# MEDGEMMA NSCLC Multimodal Processing Script
# ---------------------------------------------------------------------
# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
# import requests
# import torch
# model_id = "google/medgemma-4b-pt"
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# processor = AutoProcessor.from_pretrained(model_id)
# # Image attribution: Stillwaterising, CC0, via Wikimedia Commons
# image_url = (
#     "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
# )
# image = Image.open(
#     requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw
# ).convert("RGB")
# prompt = "<start_of_image> findings:"
# inputs = processor(text=prompt, images=image, return_tensors="pt").to(
#     model.device, dtype=torch.bfloat16
# )
# # Extract embeddings instead of generating text
# with torch.inference_mode():
#     # Get model outputs with return_dict=True to access hidden states
#     outputs = model(**inputs, output_hidden_states=True, return_dict=True)
#     print(f"Output keys: {outputs.keys()}") 
#     # Get the last hidden state (embeddings)
#     last_hidden_state = outputs.hidden_states[-1]
#     # Get the embeddings for the entire sequence
#     embeddings = last_hidden_state
#     # Alternatively, you can get the embeddings for just the [EOS] token
#     # which often represents the whole sequence
#     # eos_embedding = last_hidden_state[0, -1, :]
# print(f"Embedding shape: {embeddings.shape}")
# print(f"Sample embedding values: {embeddings[0, 0, :10]}")  # Print first 10 values
# # You can also get the pooled representation if needed
# # Average pooling across token dimension
# pooled_embedding = torch.mean(embeddings, dim=1)
# print(f"Pooled embedding shape: {pooled_embedding.shape}") # 2560

import os
import pydicom
import numpy as np
import cv2
import torch
import pandas as pd
import torch.multiprocessing as mp
import multiprocessing
import gc
import traceback
import zipfile
from io import BytesIO
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Update model to MedGemma
model_id = "google/medgemma-4b-pt"

# Update paths for NSCLC data
BASE_PATH = "/roshare/dept_machinelearning/Faculty/Rasool, Ghulam"

# Data paths
NSCLC_PATH = f"{BASE_PATH}/Shared Resources/CDSC_Data_Sets/Image-Data-Zipped/10R24000015"
MANIFEST_PATH = "/proj/rasool_lab_projects/Aakash/ARCHIVE/private/NSCLC/data/manifest.csv"
CANCER_REGISTRY_PATH = "/proj/rasool_lab_projects/Aakash/ARCHIVE/private/NSCLC/data/cancer_registry_cleaned.csv"

# Output paths
OUTPUT_DIR = "/proj/rasool_lab_projects/Aakash/MEDGEMMA/data/"
OUTPUT_FILE = "MedGemma-NSCLC.parquet"

def create_clinical_text_prompt(clinical_data, clinical_cols, diagnosis_cols):
    """Create a structured clinical text prompt for multimodal embedding"""
    text_parts = []
    
    # Add clinical information
    clinical_info = []
    for col in clinical_cols:
        if col in clinical_data and pd.notna(clinical_data[col]) and str(clinical_data[col]).strip():
            # Clean up column names for better readability
            clean_col = col.replace('_', ' ').replace(' DESC', '').replace(' NUM', '').title()
            clinical_info.append(f"{clean_col}: {str(clinical_data[col]).strip()}")
    
    if clinical_info:
        text_parts.append("Clinical Information: " + " | ".join(clinical_info))
    
    # Add diagnosis information
    diagnosis_info = []
    for col in diagnosis_cols:
        if col in clinical_data and pd.notna(clinical_data[col]) and str(clinical_data[col]).strip():
            clean_col = col.replace('_TXT', '').replace('DX_PROC_', '').replace('_', ' ').title()
            diagnosis_info.append(f"{clean_col}: {str(clinical_data[col]).strip()}")
    
    if diagnosis_info:
        text_parts.append("Diagnostic Information: " + " | ".join(diagnosis_info))
    
    # Create final prompt
    if text_parts:
        clinical_text = " || ".join(text_parts)
        return f"<start_of_image> Patient clinical data: {clinical_text}. Radiological findings:"
    else:
        return "<start_of_image> findings:"

class DICOMPreprocessor:
    def __init__(
        self,
        target_size=None,
        window_center=None,
        window_width=None,
        normalize=True,
        to_three_channels=True,
    ):
        self.target_size = target_size
        self.window_center = window_center
        self.window_width = window_width
        self.normalize = normalize
        self.to_three_channels = to_three_channels

    def load_dicom(self, dicom_data):
        """Load DICOM from file path or bytes"""
        try:
            if isinstance(dicom_data, str):
                dicom = pydicom.dcmread(dicom_data)
            else:
                dicom = pydicom.dcmread(dicom_data)

            if (
                hasattr(dicom, "PixelData")
                or hasattr(dicom, "FloatPixelData")
                or hasattr(dicom, "DoubleFloatPixelData")
            ):
                return dicom.pixel_array.astype(np.float32)
            else:
                return None
        except Exception as e:
            return None

    def apply_windowing(self, image):
        if self.window_center is not None and self.window_width is not None:
            min_value = self.window_center - self.window_width // 2
            max_value = self.window_center + self.window_width // 2
            image = np.clip(image, min_value, max_value)
            image = (image - min_value) / (max_value - min_value)
        return image

    def resize_image(self, image):
        if self.target_size is not None:
            return cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return image

    def normalize_image(self, image):
        if self.normalize:
            min_val = np.min(image)
            max_val = np.max(image)
            return (
                (image - min_val) / (max_val - min_val) if max_val > min_val else image
            )
        return image

    def convert_to_three_channels(self, image):
        if self.to_three_channels and image.ndim < 3:
            return np.stack((image,) * 3, axis=-1)
        return image

    def preprocess(self, dicom_data):
        image = self.load_dicom(dicom_data)
        if image is None:
            return None
        image = self.apply_windowing(image)
        image = self.normalize_image(image)
        image = self.resize_image(image)
        image = self.convert_to_three_channels(image)
        return image


class MedGemmaMultimodalEncoder:
    def __init__(self, device_id: int = None) -> None:
        self.model = None
        self.processor = None

        # Configure device properly
        if device_id is not None and torch.cuda.is_available():
            try:
                self.device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device_id)
            except Exception as e:
                print(f"Failed to set GPU {device_id}: {str(e)}")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self._load_model()

    def _load_model(self) -> None:
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto" if "cuda" in str(self.device) else "cpu"
            )
            self.model.eval()

            # Force CUDA initialization
            if "cuda" in str(self.device):
                with torch.no_grad():
                    dummy_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
                    dummy_inputs = self.processor(
                        text="<start_of_image> findings:", 
                        images=dummy_image, 
                        return_tensors="pt"
                    ).to(self.device, dtype=torch.bfloat16)
                    _ = self.model(**dummy_inputs, output_hidden_states=True, return_dict=True)
                    torch.cuda.synchronize()

        except Exception as e:
            print(f"Error loading MedGemma model: {str(e)}")
            traceback.print_exc()
            raise

    def encode_multimodal(self, images_list, clinical_text_prompt) -> torch.Tensor:
        """Encode images with clinical text using MedGemma multimodal capabilities"""
        if self.model is None or self.processor is None:
            raise ValueError("Model is not loaded.")

        # Process in smaller batches to avoid memory issues
        batch_size = 4  # Reduced for multimodal processing
        all_embeddings = []
        
        try:
            for i in range(0, len(images_list), batch_size):
                batch_images = images_list[i:i+batch_size]
                batch_embeddings = []
                
                for image in batch_images:
                    # Convert numpy array to PIL Image if needed
                    if isinstance(image, np.ndarray):
                        # Normalize to 0-255 range and convert to uint8
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                        
                        # Handle different array shapes
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            pil_image = Image.fromarray(image, mode='RGB')
                        elif len(image.shape) == 2:
                            pil_image = Image.fromarray(image, mode='L').convert('RGB')
                        else:
                            # If it's (3, H, W), transpose to (H, W, 3)
                            if image.shape[0] == 3:
                                image = np.transpose(image, (1, 2, 0))
                            pil_image = Image.fromarray(image, mode='RGB')
                    else:
                        pil_image = image
                    
                    print(f"Processing image {i + len(batch_embeddings) + 1}/{len(images_list)}")
                    print(f"Image size: {pil_image.size}, Mode: {pil_image.mode}")

                    # Process with MedGemma using both image and clinical text
                    inputs = self.processor(
                        text=clinical_text_prompt, 
                        images=pil_image, 
                        return_tensors="pt"
                    ).to(self.device, dtype=torch.bfloat16)
                    
                    # Clear cache before inference
                    if "cuda" in str(self.device):
                        torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                        last_hidden_state = outputs.hidden_states[-1]
                        # Use average pooling across token dimension for multimodal representation
                        pooled_embedding = torch.mean(last_hidden_state, dim=1)
                        
                        if "cuda" in str(self.device):
                            torch.cuda.synchronize()
                        
                        batch_embeddings.append(pooled_embedding.cpu().float())
                
                if batch_embeddings:
                    batch_tensor = torch.cat(batch_embeddings, dim=0)
                    all_embeddings.append(batch_tensor)

            # Concatenate all batches
            if all_embeddings:
                return torch.cat(all_embeddings, dim=0).numpy()
            else:
                return np.array([])

        except RuntimeError as e:
            print(f"CUDA error during inference: {str(e)}")
            # Fallback to CPU
            self.model = self.model.cpu()
            return self.encode_multimodal(images_list, clinical_text_prompt)


def extract_dicom_from_zip(zip_path, dcm_file):
    """Extract specific DICOM file from ZIP archive"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            try:
                dcm_data = BytesIO(zip_ref.read(dcm_file))
                return dcm_data
            except KeyError:
                file_list = zip_ref.namelist()
                matching_files = [
                    f for f in file_list if f.endswith(dcm_file.split("/")[-1])
                ]

                if matching_files:
                    dcm_data = BytesIO(zip_ref.read(matching_files[0]))
                    return dcm_data
                else:
                    dicom_files = [f for f in file_list if f.lower().endswith(".dcm")]
                    if dicom_files:
                        dcm_data = BytesIO(zip_ref.read(dicom_files[0]))
                        return dcm_data
                    else:
                        return None
    except Exception as e:
        return None


def extract_all_dicoms_from_zip(zip_path):
    """Extract all DICOM files from ZIP archive"""
    dicom_slices = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            dicom_files = [f for f in zip_ref.namelist() if f.lower().endswith(".dcm")]

            for dcm_file in dicom_files:
                try:
                    dcm_data = BytesIO(zip_ref.read(dcm_file))
                    dicom_slices.append(dcm_data)
                except Exception as e:
                    continue

            return dicom_slices
    except Exception as e:
        return []


def worker_process(gpu_id, task_queue, result_queue, preprocessor_args, clinical_data_dict):
    """Worker process to process tasks from the task queue"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    
    # Load the model ONCE per worker process
    encoder = MedGemmaMultimodalEncoder(device_id=gpu_id)
    print(f"Worker {gpu_id}: Model loaded successfully")
    
    while True:
        task = task_queue.get()
        if task is None:
            break
        result = process_patient_scan_multimodal(task, preprocessor_args, clinical_data_dict, encoder)
        result_queue.put(result)


def process_patient_scan_multimodal(row, preprocessor_args, clinical_data_dict, encoder):
    """Process a patient scan using MedGemma multimodal capabilities"""
    try:
        # Remove the encoder creation - it's now passed as a parameter
        # encoder = MedGemmaMultimodalEncoder(device_id=gpu_id)  # REMOVE THIS LINE

        # Convert all values to strings
        patient_id = str(row["patient_id"])
        session = str(row["session"])
        acquisition = str(row["acquisition"])
        zip_file = str(row["zip_file"])
        dcm_file = str(row["dcm_file"])
        study_description = str(row.get("StudyDescription", ""))

        # Get clinical data for this patient
        clinical_data = clinical_data_dict.get(patient_id, {})
        
        # Define clinical and diagnosis columns
        clinical_cols = [
            'AGE_AT_DIAGNOSIS_NUM',
            'RACE_CR_SRC_DESC_1',
            'ETHNICITY_SRC_DESC',
            'HISTOLOGY_DESC',
            'STAGE_CLINICAL_TNM_T_DESC',
            'STAGE_CLINICAL_TNM_N_DESC',
            'STAGE_CLINICAL_TNM_M_DESC',
            'STAGE_PATHOLOGICAL_TNM_T_DESC',
            'STAGE_PATHOLOGICAL_TNM_N_DESC',
            'STAGE_PATHOLOGICAL_TNM_M_DESC',
            'DERIVED_TOBACCO_SMOKING_STATUS_DESC',
        ]

        diagnosis_cols = [
            'DX_PROC_LAB_TESTS_TXT',
            'DX_PROC_PHYSICAL_EXAM_TXT',
            'DX_PROC_PATH_TXT',
            'DX_PROC_SCOPES_TXT',
            'DX_PROC_XRAY_SCAN_TXT',
            'REMARKS_TXT',
        ]
        
        # Create clinical text prompt
        clinical_text_prompt = create_clinical_text_prompt(clinical_data, clinical_cols, diagnosis_cols)

        # Construct the full path to the ZIP file
        zip_path = os.path.join(
            NSCLC_PATH,
            "SUBJECTS",
            patient_id,
            "SESSIONS",
            session,
            "ACQUISITIONS",
            acquisition,
            "FILES",
            zip_file,
        )

        if not os.path.exists(zip_path):
            return None

        # Extract all DICOM slices
        dicom_slices = extract_all_dicoms_from_zip(zip_path)

        # Create preprocessor
        preprocessor = DICOMPreprocessor(**preprocessor_args)

        # Process all slices
        processed_images = []
        for dcm_data in dicom_slices:
            image = preprocessor.preprocess(dcm_data)
            if image is not None:
                processed_images.append(image)

        if processed_images:
            # Encode images with clinical text using MedGemma multimodal
            multimodal_embeddings = encoder.encode_multimodal(processed_images, clinical_text_prompt)

            # Return patient data with multimodal embeddings
            result = {
                "patient_id": patient_id,
                "session": session,
                "acquisition": acquisition,
                "zip_file": zip_file,
                "dcm_file": dcm_file,
                "source_type": row.get("source_type", ""),
                "study_description": study_description,
                "multimodal_embeddings": multimodal_embeddings,
                "clinical_text_used": clinical_text_prompt,
            }

            # Force garbage collection (but don't delete the encoder!)
            del multimodal_embeddings, processed_images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result

        return None

    except Exception as e:
        print(f"Error processing patient {row.get('patient_id', 'unknown')}: {str(e)}")
        return None


def aggregate_multimodal_embeddings_by_patient(results_df):
    """Aggregate multimodal embeddings by patient and contrast type"""
    aggregated_data = []
    
    # Group by patient_id
    for patient_id, patient_group in results_df.groupby('patient_id'):
        # Initialize dictionaries to store embeddings by contrast type
        contrast_embeddings = []
        wo_contrast_embeddings = []
        
        # Process each scan for this patient
        for _, row in patient_group.iterrows():
            study_desc = row['study_description'].lower()
            embeddings = row['multimodal_embeddings']
            
            # Categorize by contrast type
            if 'w contrast' in study_desc or 'with contrast' in study_desc:
                contrast_embeddings.append(embeddings)
            elif 'wo contrast' in study_desc or 'without contrast' in study_desc:
                wo_contrast_embeddings.append(embeddings)
        
        # Aggregate embeddings for each contrast type
        patient_data = {
            'patient_id': patient_id,
            'source_type': patient_group['source_type'].iloc[0],
            'clinical_text_used': patient_group['clinical_text_used'].iloc[0],
        }
        
        # Concatenate contrast embeddings if any exist
        if contrast_embeddings:
            all_contrast_emb = np.concatenate(contrast_embeddings, axis=0)
            patient_data['multimodal_contrast_embeddings'] = all_contrast_emb.tobytes()
            patient_data['multimodal_contrast_embedding_shape'] = all_contrast_emb.shape
        else:
            patient_data['multimodal_contrast_embeddings'] = None
            patient_data['multimodal_contrast_embedding_shape'] = None
        
        # Concatenate without contrast embeddings if any exist
        if wo_contrast_embeddings:
            all_wo_contrast_emb = np.concatenate(wo_contrast_embeddings, axis=0)
            patient_data['multimodal_wo_contrast_embeddings'] = all_wo_contrast_emb.tobytes()
            patient_data['multimodal_wo_contrast_embedding_shape'] = all_wo_contrast_emb.shape
        else:
            patient_data['multimodal_wo_contrast_embeddings'] = None
            patient_data['multimodal_wo_contrast_embedding_shape'] = None
        
        aggregated_data.append(patient_data)
    
    return pd.DataFrame(aggregated_data)


if __name__ == "__main__":
    # Read the manifest file
    manifest_df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest with {len(manifest_df)} entries")
    
    # Load cancer registry data to get clinical information
    print("Loading clinical data...")
    cr_df = pd.read_csv(CANCER_REGISTRY_PATH)
    
    # Create a dictionary for quick lookup of clinical data by patient ID
    clinical_data_dict = {}
    for _, row in cr_df.iterrows():
        patient_id = str(row['MRN'])
        clinical_data_dict[patient_id] = row.to_dict()
    
    print(f"Loaded clinical data for {len(clinical_data_dict)} patients")
    
    # Check StudyDescription distribution
    if 'StudyDescription' in manifest_df.columns:
        print("\nStudyDescription distribution:")
        print(manifest_df['StudyDescription'].value_counts())
    else:
        print("Warning: StudyDescription column not found in manifest")

    # Create tasks from manifest rows
    tasks = manifest_df.to_dict("records")

    preprocessor_args = {
        "target_size": 224,
        "normalize": True,
        "to_three_channels": True,
    }

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus == 0:
        print("No GPUs found. Using CPU only.")
        # Process sequentially on CPU - load model once
        encoder = MedGemmaMultimodalEncoder(device_id=None)
        all_results = []
        for task in tqdm(tasks, desc="Processing"):
            result = process_patient_scan_multimodal(task, preprocessor_args, clinical_data_dict, encoder)
            if result:
                all_results.append(result)
    else:
        # Set up multiprocessing with proper initialization
        try:
            # Initialize multiprocessing with spawn method for CUDA compatibility
            mp.set_start_method("spawn", force=True)

            # Create queues for task distribution and result collection
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()

            # Put all tasks in the queue
            for task in tasks:
                task_queue.put(task)

            # Add termination signals (one per GPU)
            for _ in range(num_gpus):
                task_queue.put(None)

            # Create worker processes
            processes = []
            for gpu_id in range(num_gpus):
                p = multiprocessing.Process(
                    target=worker_process,
                    args=(gpu_id, task_queue, result_queue, preprocessor_args, clinical_data_dict),
                )
                processes.append(p)
                p.start()

            print(f"Started {num_gpus} worker processes")

            # Collect results
            all_results = []
            num_results = 0
            expected_results = len(tasks)

            # Collect results as they come in
            with tqdm(total=expected_results, desc="Processing patients") as pbar:
                while any(p.is_alive() for p in processes):
                    try:
                        result = result_queue.get(timeout=5)
                        all_results.append(result)
                        num_results += 1
                        pbar.update(1)
                    except multiprocessing.queues.Empty:
                        continue

                # Get any remaining results
                while not result_queue.empty():
                    result = result_queue.get()
                    all_results.append(result)
                    num_results += 1
                    pbar.update(1)

            # Wait for all processes to complete
            for i, p in enumerate(processes):
                p.join(timeout=60)
                if p.is_alive():
                    print(f"Worker on GPU {i} did not terminate properly, killing it")
                    p.kill()

            print(f"Completed: {num_results}/{expected_results} patients processed")

        except Exception as e:
            print(f"Error in multiprocessing: {str(e)}")
            # Fallback to sequential processing
            print("Falling back to sequential processing")
            all_results = []
            for task in tqdm(tasks, desc="Processing"):
                result = process_patient_scan_multimodal(task, preprocessor_args, clinical_data_dict, 0)
                if result:
                    all_results.append(result)

    # Convert the data into a Pandas DataFrame
    valid_results = [result for result in all_results if result is not None]
    results_df = pd.DataFrame(valid_results)
    print(f"Individual scans processed: {len(results_df)} (out of {len(all_results)} total attempts)")
    
    # Aggregate multimodal embeddings by patient and contrast type
    print("Aggregating multimodal embeddings by patient and contrast type...")
    aggregated_df = aggregate_multimodal_embeddings_by_patient(results_df)
    print(f"Unique patients after aggregation: {len(aggregated_df)}")
    
    # Print statistics about contrast types
    contrast_count = aggregated_df['multimodal_contrast_embeddings'].notna().sum()
    wo_contrast_count = aggregated_df['multimodal_wo_contrast_embeddings'].notna().sum()
    both_count = (aggregated_df['multimodal_contrast_embeddings'].notna() & 
                  aggregated_df['multimodal_wo_contrast_embeddings'].notna()).sum()
    
    print(f"Patients with contrast CT multimodal embeddings: {contrast_count}")
    print(f"Patients with non-contrast CT multimodal embeddings: {wo_contrast_count}")
    print(f"Patients with both types: {both_count}")

    # Load survival data and merge
    print("Merging with survival data...")
    
    # Select survival columns
    survival_cols = [
        'MRN', 
        'SURVIVAL_TIME_IN_MONTHS', 
        'VITAL_STATUS_DESC',
    ]

    # Prepare survival data
    survival_data = cr_df[survival_cols].copy()
    survival_data['event'] = (survival_data['VITAL_STATUS_DESC'] == 'DEAD').astype(int)
    survival_data = survival_data.rename(columns={'MRN': 'patient_id'})
    
    # Ensure patient_id columns are string type for merging
    aggregated_df['patient_id'] = aggregated_df['patient_id'].astype(str)
    survival_data['patient_id'] = survival_data['patient_id'].astype(str)

    # Merge with survival data
    final_df = aggregated_df.merge(
        survival_data, 
        on='patient_id',
        how='left'
    )

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save DataFrame to Parquet file
    output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    final_df.to_parquet(output_file_path, engine="pyarrow")

    print(f"Multimodal data saved to {output_file_path}")

    # Save the final merged data
    final_output_path = "/proj/rasool_lab_projects/Aakash/EAGLE/data/NSCLC_MedGemma_Multimodal.parquet"
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    final_df.to_parquet(final_output_path)
    
    print(f"Final multimodal dataset saved to {final_output_path}")
    print(f"Total patients with multimodal embeddings: {len(aggregated_df)}")
    print(f"Total patients with survival data: {len(survival_data)}")
    print(f"Total patients in merged dataset: {len(final_df)}")
    print(f"Patients with both multimodal and survival data: {final_df['SURVIVAL_TIME_IN_MONTHS'].notna().sum()}")