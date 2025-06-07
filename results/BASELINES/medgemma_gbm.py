# MEDGEMMA GBM Multimodal Processing Script
# ---------------------------------------------------------------------
import os
import json
import numpy as np
import nibabel as nib
import cv2
import torch
import pandas as pd
import torch.multiprocessing as mp
import multiprocessing
import gc
import traceback
import psutil
from io import BytesIO
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from datetime import datetime

# Update model to MedGemma
model_id = "google/medgemma-4b-pt"

# Update paths for GBM data
BASE_PATH = "/roshare/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Iryna/Neuro-dataset/FL-GBM-Data/FL-GBM-MRIs-preprocessed/"
MANIFEST_PATH = "/proj/rasool_lab_projects/Aakash/ARCHIVE/private/GBM/data/manifest.json"

# Output paths
OUTPUT_DIR = "/proj/rasool_lab_projects/Aakash/MEDGEMMA/data/"
OUTPUT_FILE = "MedGemma-GBM.parquet"

# Co-registered MRI sequence types to process (brain masked)
MRI_SEQUENCES = {
    "FL": "FL_to_SRI_brain.nii.gz",
    "T1": "T1_to_SRI_brain.nii.gz", 
    "T1CE": "T1CE_to_SRI_brain.nii.gz",
    "T2": "T2_to_SRI_brain.nii.gz"
}

def print_header(title, width=80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width)

def print_section(title, width=80):
    """Print a formatted section header"""
    print("\n" + "-" * width)
    print(f" {title} ")
    print("-" * width)

def print_status(message, level="INFO"):
    """Print a formatted status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{timestamp}] [{level}]"
    print(f"{prefix} {message}")

def print_gpu_info():
    """Print GPU information in a clean format"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print_section("GPU Information")
        print(f"📊 Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
    else:
        print_section("GPU Information")
        print("⚠️  No CUDA GPUs available - using CPU")

def create_clinical_text_prompt(patient_manifest):
    """Create a structured clinical text prompt for multimodal embedding"""
    text_parts = []
    
    # Clinical information
    clinical_info = []
    clinical_fields = [
        ("age_at_diagnosis_num", "Age at Diagnosis"),
        ("gender_src_desc", "Gender"),
        ("race_cr_src_desc_1", "Race"),
        ("ethnicity_src_desc", "Ethnicity"),
        ("histology_txt", "Histology"),
        ("clinical_tumor_size", "Tumor Size"),
        ("regional_nodes_positive", "Regional Nodes Positive"),
        ("tnm_path_stagedby_desc", "TNM Staging"),
        ("derived_tobacco_smoking_status_desc", "Smoking Status"),
        ("patient_height", "Height"),
        ("patient_weight", "Weight")
    ]
    
    for field, label in clinical_fields:
        value = patient_manifest.get(field)
        if value and str(value).strip() and str(value) != "None":
            clinical_info.append(f"{label}: {str(value).strip()}")
    
    if clinical_info:
        text_parts.append("Clinical Information: " + " | ".join(clinical_info))
    
    # Add radiology report if available
    radiology_report = patient_manifest.get("radiology_report", "")
    if radiology_report and radiology_report.strip():
        # Truncate very long reports to avoid token limits
        if len(radiology_report) > 500:
            radiology_report = radiology_report[:500] + "..."
        text_parts.append(f"Radiology Report: {radiology_report.strip()}")
    
    # Add pathology report if available  
    pathology_report = patient_manifest.get("pathology_report", "")
    if pathology_report and pathology_report.strip():
        # Truncate very long reports
        if len(pathology_report) > 500:
            pathology_report = pathology_report[:500] + "..."
        text_parts.append(f"Pathology Report: {pathology_report.strip()}")
    
    # Add treatment information if available
    treatment = patient_manifest.get("treatment", "")
    if treatment and treatment.strip():
        if len(treatment) > 300:
            treatment = treatment[:300] + "..."
        text_parts.append(f"Treatment: {treatment.strip()}")
    
    # Create final prompt
    if text_parts:
        clinical_text = " || ".join(text_parts)
        return f"<start_of_image> Patient clinical data: {clinical_text}. Brain MRI findings:"
    else:
        return "<start_of_image> Brain MRI findings:"

class NIfTIPreprocessor:
    def __init__(
        self,
        target_size=None,
        normalize=True,
        to_three_channels=True,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.to_three_channels = to_three_channels

    def load_nifti(self, nifti_path):
        """Load NIfTI file and return 3D volume"""
        try:
            if not os.path.exists(nifti_path):
                return None
            nifti_data = nib.load(nifti_path)
            data = nifti_data.get_fdata().astype(np.float32)
            return data
        except Exception as e:
            print_status(f"Error loading NIfTI file {nifti_path}: {str(e)}", "ERROR")
            return None

    def normalize_image(self, image):
        if self.normalize:
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val)
            return image
        return image

    def resize_image(self, image):
        if self.target_size is not None:
            return cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return image

    def convert_to_three_channels(self, image):
        if self.to_three_channels and image.ndim < 3:
            return np.stack((image,) * 3, axis=-1)
        return image

    def preprocess_slice(self, slice_data):
        """Process a single 2D slice from a 3D volume"""
        if slice_data is None:
            return None
        slice_data = self.normalize_image(slice_data)
        slice_data = self.resize_image(slice_data)
        slice_data = self.convert_to_three_channels(slice_data)
        return slice_data

    def preprocess_volume(self, nifti_path):
        """Process an entire 3D NIfTI volume and return slices"""
        volume = self.load_nifti(nifti_path)
        if volume is None:
            return None

        # Get the number of slices in the appropriate dimension
        if volume.ndim >= 3:
            num_slices = volume.shape[2]
            processed_slices = []

            # Sample slices evenly across the volume to reduce memory usage
            # Take every nth slice to get approximately 30-40 slices
            step = max(1, num_slices // 35)
            
            for i in range(0, num_slices, step):
                slice_data = volume[:, :, i]
                # Skip empty slices
                if np.max(slice_data) > 0:
                    processed_slice = self.preprocess_slice(slice_data)
                    if processed_slice is not None:
                        processed_slices.append(processed_slice)

            return processed_slices
        else:
            return None


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
        batch_size = 2  # Reduced for multimodal processing with MRI
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
            self.device = torch.device("cpu")
            return self.encode_multimodal(images_list, clinical_text_prompt)


def find_coregistered_mri_files(patient_manifest):
    """Find all co-registered MRI files for a patient from the manifest"""
    coregistered_files = {}
    
    if "MRIs" not in patient_manifest:
        return coregistered_files
    
    for mri_session in patient_manifest["MRIs"]:
        if mri_session.get("is_coregistered", False) and "coregistered_files" in mri_session:
            for file_info in mri_session["coregistered_files"]:
                sequence_type = file_info.get("sequence_type", "")
                file_path = file_info.get("path", "")
                
                if sequence_type and file_path and os.path.exists(file_path):
                    if sequence_type not in coregistered_files:
                        coregistered_files[sequence_type] = file_path
    
    return coregistered_files


def worker_process(gpu_id, task_queue, result_queue, preprocessor_args):
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
        result = process_patient_multimodal(task, preprocessor_args, encoder)
        result_queue.put(result)


def process_patient_multimodal(patient_manifest, preprocessor_args, encoder):
    """Process a patient using MedGemma multimodal capabilities"""
    try:
        patient_id = patient_manifest.get("patient_id")
        
        # Create clinical text prompt from patient data
        clinical_text_prompt = create_clinical_text_prompt(patient_manifest)
        
        # Find co-registered MRI files
        coregistered_files = find_coregistered_mri_files(patient_manifest)
        
        if not coregistered_files:
            print_status(f"No co-registered MRI files found for patient {patient_id}", "WARN")
            return None
        
        # Create preprocessor
        preprocessor = NIfTIPreprocessor(**preprocessor_args)
        
        # Process each MRI sequence and collect all slices
        sequence_embeddings = {}
        
        for sequence_type, file_path in coregistered_files.items():
            print_status(f"Processing {sequence_type} sequence for patient {patient_id}", "SEQ")
            
            # Process the NIfTI volume
            processed_slices = preprocessor.preprocess_volume(file_path)
            
            if processed_slices and len(processed_slices) > 0:
                # Encode slices with clinical text using MedGemma multimodal
                multimodal_embeddings = encoder.encode_multimodal(processed_slices, clinical_text_prompt)
                
                if multimodal_embeddings.size > 0:
                    sequence_embeddings[sequence_type] = multimodal_embeddings
                    print_status(f"✓ {sequence_type} processed - shape: {multimodal_embeddings.shape}", "SUCCESS")
                
                # Force garbage collection
                del processed_slices
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if not sequence_embeddings:
            return None
        
        # Prepare result with patient data and multimodal embeddings by sequence
        result = {
            "patient_id": patient_id,
            "clinical_text_used": clinical_text_prompt,
        }
        
        # Add clinical fields from manifest
        clinical_fields = [
            "survival_time_in_months", "vital_status_desc", "dx_dt", "age_at_diagnosis_num",
            "gender_src_desc", "race_cr_src_desc_1", "ethnicity_src_desc", 
            "histology_txt", "clinical_tumor_size", "regional_nodes_positive", 
            "tnm_path_stagedby_desc", "derived_tobacco_smoking_status_desc", 
            "patient_height", "patient_weight", "mrn", "date"
        ]
        
        for field in clinical_fields:
            result[field] = patient_manifest.get(field)
        
        # Store multimodal embeddings for each sequence type
        for sequence_type, embeddings in sequence_embeddings.items():
            result[f"{sequence_type}_embeddings"] = embeddings
        
        # Add summary information
        result["total_sequences_processed"] = len(sequence_embeddings)
        result["available_sequences"] = ",".join(sequence_embeddings.keys())
        
        print_status(f"✓ Patient {patient_id} completed - {len(sequence_embeddings)}/{len(coregistered_files)} sequences processed", "SUCCESS")
        return result
        
    except Exception as e:
        print(f"Error processing patient {patient_manifest.get('patient_id', 'unknown')}: {str(e)}")
        traceback.print_exc()
        return None


def aggregate_multimodal_embeddings_by_patient(results_df):
    """Aggregate multimodal embeddings by patient and MRI sequence type"""
    aggregated_data = []
    
    # Process each patient
    for _, row in results_df.iterrows():
        patient_data = {
            'patient_id': row['patient_id'],
            'clinical_text_used': row['clinical_text_used'],
        }
        
        # Copy clinical fields
        clinical_fields = [
            "survival_time_in_months", "vital_status_desc", "dx_dt", "age_at_diagnosis_num",
            "gender_src_desc", "race_cr_src_desc_1", "ethnicity_src_desc", 
            "histology_txt", "clinical_tumor_size", "regional_nodes_positive", 
            "tnm_path_stagedby_desc", "derived_tobacco_smoking_status_desc", 
            "patient_height", "patient_weight", "mrn", "date"
        ]
        
        for field in clinical_fields:
            if field in row:
                patient_data[field] = row[field]
        
        # Process each sequence type
        sequence_types = ['FL', 'T1', 'T1CE', 'T2']
        available_sequences = []
        
        for seq_type in sequence_types:
            col_name = f"{seq_type}_embeddings"
            if col_name in row and row[col_name] is not None:
                embeddings = row[col_name]
                if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
                    patient_data[f'{seq_type}_embeddings'] = embeddings.astype(np.float32).tobytes()
                    patient_data[f'{seq_type}_embedding_shape'] = embeddings.shape
                    available_sequences.append(seq_type)
                else:
                    patient_data[f'{seq_type}_embeddings'] = None
                    patient_data[f'{seq_type}_embedding_shape'] = None
            else:
                patient_data[f'{seq_type}_embeddings'] = None
                patient_data[f'{seq_type}_embedding_shape'] = None
        
        # Create combined multimodal embedding by pooling across all available sequences
        all_sequence_embeddings = []
        for seq_type in available_sequences:
            embeddings = row[f"{seq_type}_embeddings"]
            if isinstance(embeddings, np.ndarray):
                all_sequence_embeddings.append(embeddings)
        
        if all_sequence_embeddings:
            # Find minimum number of slices across sequences
            min_slices = min(emb.shape[0] for emb in all_sequence_embeddings)
            
            # Truncate all sequences to same number of slices
            aligned_embeddings = []
            for emb in all_sequence_embeddings:
                aligned_embeddings.append(emb[:min_slices])
            
            # Stack and mean pool across sequences
            stacked_embeddings = np.stack(aligned_embeddings, axis=0)
            combined_embedding = np.mean(stacked_embeddings, axis=0)
            
            patient_data['combined_embeddings'] = combined_embedding.astype(np.float32).tobytes()
            patient_data['combined_embedding_shape'] = combined_embedding.shape
            patient_data['num_slices'] = min_slices
        else:
            patient_data['combined_embeddings'] = None
            patient_data['combined_embedding_shape'] = None
            patient_data['num_slices'] = 0
        
        patient_data['available_sequences'] = ",".join(available_sequences)
        patient_data['total_sequences_available'] = len(available_sequences)
        
        aggregated_data.append(patient_data)
    
    return pd.DataFrame(aggregated_data)


if __name__ == "__main__":
    # Print startup header
    print_header("GBM MedGemma Multimodal Processing Pipeline")
    print_status("Starting GBM multimodal processing pipeline", "INIT")
    
    # Print system information
    print_gpu_info()
    
    # Load manifest
    print_section("Loading Data")
    print_status("Loading manifest file...", "LOAD")
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    
    # Filter patients with co-registered MRI data
    print_status("Filtering patients with co-registered MRI data...", "FILTER")
    valid_patients = []
    for patient in manifest:
        has_coregistered_mri = False
        if "MRIs" in patient and patient["MRIs"]:
            for mri_session in patient["MRIs"]:
                if mri_session.get("is_coregistered", False):
                    has_coregistered_mri = True
                    break
        
        if has_coregistered_mri:
            valid_patients.append(patient)
    
    manifest = valid_patients
    print(f"Loaded manifest with {len(manifest)} patients having co-registered MRI data")
    
    # Setup preprocessing parameters
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
        for patient in tqdm(manifest, desc="Processing"):
            result = process_patient_multimodal(patient, preprocessor_args, encoder)
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
            for patient in manifest:
                task_queue.put(patient)

            # Add termination signals (one per GPU)
            for _ in range(num_gpus):
                task_queue.put(None)

            # Create worker processes
            processes = []
            for gpu_id in range(num_gpus):
                p = multiprocessing.Process(
                    target=worker_process,
                    args=(gpu_id, task_queue, result_queue, preprocessor_args),
                )
                processes.append(p)
                p.start()

            print(f"Started {num_gpus} worker processes")

            # Collect results
            all_results = []
            num_results = 0
            expected_results = len(manifest)

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
            encoder = MedGemmaMultimodalEncoder(device_id=0)
            all_results = []
            for patient in tqdm(manifest, desc="Processing"):
                result = process_patient_multimodal(patient, preprocessor_args, encoder)
                if result:
                    all_results.append(result)

    # Convert the data into a Pandas DataFrame
    valid_results = [result for result in all_results if result is not None]
    results_df = pd.DataFrame(valid_results)
    print(f"Individual patients processed: {len(results_df)} (out of {len(all_results)} total attempts)")
    
    # Aggregate multimodal embeddings by patient and sequence type
    print("Aggregating multimodal embeddings by patient and sequence type...")
    aggregated_df = aggregate_multimodal_embeddings_by_patient(results_df)
    print(f"Unique patients after aggregation: {len(aggregated_df)}")
    
    # Print statistics about sequence types
    sequence_types = ['FL', 'T1', 'T1CE', 'T2']
    for seq_type in sequence_types:
        count = aggregated_df[f'{seq_type}_embeddings'].notna().sum()
        print(f"Patients with {seq_type} multimodal embeddings: {count}")
    
    combined_count = aggregated_df['combined_embeddings'].notna().sum()
    print(f"Patients with combined multimodal embeddings: {combined_count}")
    
    # Add survival event indicator
    aggregated_df['event'] = (aggregated_df['vital_status_desc'] == 'DEAD').astype(int)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save DataFrame to Parquet file
    output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    aggregated_df.to_parquet(output_file_path, engine="pyarrow")

    print(f"Multimodal data saved to {output_file_path}")

    # Save the final merged data
    final_output_path = "/proj/rasool_lab_projects/Aakash/EAGLE/data/GBM_MedGemma_Multimodal.parquet"
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    aggregated_df.to_parquet(final_output_path)
    
    print(f"Final multimodal dataset saved to {final_output_path}")
    print(f"Total patients with multimodal embeddings: {len(aggregated_df)}")
    print(f"Patients with survival data: {aggregated_df['survival_time_in_months'].notna().sum()}")
    
    # Final summary
    print_header("Processing Complete")
    print(f"📁 Output file: {final_output_path}")
    print(f"📊 Total patients processed: {len(aggregated_df)}")
    print(f"🧠 Patients with combined multimodal embeddings: {combined_count}")
    print(f"📈 Success rate: {len(aggregated_df)/len(manifest)*100:.1f}%")
    print(f"🔢 DataFrame shape: {aggregated_df.shape}")