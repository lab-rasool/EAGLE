# MEDGEMMA IPMN Multimodal Processing Script
# ---------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
from datasets import Dataset, concatenate_datasets
import gc
import traceback
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import psutil
import time
import pickle
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from io import BytesIO

# Update model to MedGemma
model_id = "google/medgemma-4b-pt"

# Configuration
BASE_PATH = "/roshare/dept_machinelearning/Faculty/Rasool, Ghulam"
IPMN_PATH = f"{BASE_PATH}/Shared Resources/Multimodal IPMN Data - PPFL"
OUTPUT_DIR = "/proj/rasool_lab_projects/Aakash/MEDGEMMA/data/"
OUTPUT_FILE = "MedGemma-IPMN.parquet"

# Clinical data paths
RAD_CSV = "/proj/rasool_lab_projects/Aakash/ARCHIVE/private/IPMN/data/MMFL Radiology Reports.csv"
PAT_CSV = "/proj/rasool_lab_projects/Aakash/ARCHIVE/private/IPMN/data/MMFL Pathology Reports.csv"
CLIN_CSV = "/proj/rasool_lab_projects/Aakash/ARCHIVE/private/IPMN/data/MMFL Clinical Data.csv"

# Checkpoint and chunk settings
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CHUNK_SIZE = 10  # Process this many patients at a time
SAVE_EVERY = 5   # Save progress every N chunks

def save_checkpoint(data, filename):
    """Save checkpoint data"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_parquet(checkpoint_path, index=False)
    else:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def load_checkpoint(filename):
    """Load checkpoint data"""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        if filename.endswith('.parquet'):
            return pd.read_parquet(checkpoint_path)
        else:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"⚠️  Error loading checkpoint {filename}: {e}")
        return None

def save_progress_log(stage, details):
    """Save progress log"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(CHECKPOINT_DIR, "progress_log.json")
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "details": details
    }
    
    # Load existing log or create new
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log = json.load(f)
        except:
            log = []
    else:
        log = []
    
    log.append(log_entry)
    
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

def create_clinical_text_prompt(patient_row, radiology_text="", pathology_text=""):
    """Create a structured clinical text prompt for multimodal embedding"""
    text_parts = []
    
    # Clinical information fields
    clinical_fields = [
        ("age", "Age"),
        ("gender", "Gender"),
        ("race", "Race"),
        ("ethnicity", "Ethnicity"),
        ("BMI", "BMI"),
        ("smoking_status", "Smoking Status"),
        ("alcohol_use", "Alcohol Use"),
        ("diabetes", "Diabetes"),
        ("pancreatitis_history", "Pancreatitis History"),
        ("family_history", "Family History"),
        ("CEA", "CEA Level"),
        ("CA19_9", "CA 19-9 Level"),
        ("cyst_size", "Cyst Size"),
        ("cyst_location", "Cyst Location"),
        ("main_duct_involvement", "Main Duct Involvement"),
        ("worrisome_features", "Worrisome Features"),
        ("high_risk_stigmata", "High Risk Stigmata")
    ]
    
    clinical_info = []
    for field, label in clinical_fields:
        if field in patient_row and pd.notna(patient_row[field]) and str(patient_row[field]).strip():
            value = str(patient_row[field]).strip()
            if value and value.lower() not in ['nan', 'none', 'null']:
                clinical_info.append(f"{label}: {value}")
    
    if clinical_info:
        text_parts.append("Clinical Information: " + " | ".join(clinical_info))
    
    # Add radiology report if available
    if radiology_text and radiology_text.strip():
        # Truncate very long reports to avoid token limits
        if len(radiology_text) > 500:
            radiology_text = radiology_text[:500] + "..."
        text_parts.append(f"Radiology Report: {radiology_text.strip()}")
    
    # Add pathology report if available
    if pathology_text and pathology_text.strip():
        # Truncate very long reports
        if len(pathology_text) > 500:
            pathology_text = pathology_text[:500] + "..."
        text_parts.append(f"Pathology Report: {pathology_text.strip()}")
    
    # Create final prompt
    if text_parts:
        clinical_text = " || ".join(text_parts)
        return f"<start_of_image> Patient clinical data: {clinical_text}. Pancreatic CT findings:"
    else:
        return "<start_of_image> Pancreatic CT findings:"

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

    def load_dicom(self, dicom_path):
        """Load DICOM file"""
        try:
            dicom = pydicom.dcmread(dicom_path)
            if hasattr(dicom, "PixelData"):
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

    def preprocess(self, dicom_path):
        image = self.load_dicom(dicom_path)
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


def explore_ipmn_structure(max_depth=3):
    """Explore IPMN directory structure to understand organization"""
    print(f"\nExploring IPMN directory structure at: {IPMN_PATH}")
    print("=" * 80)
    
    def print_tree(path, prefix="", depth=0):
        if depth >= max_depth:
            return
            
        try:
            items = sorted(os.listdir(path))
            dirs = [d for d in items if os.path.isdir(os.path.join(path, d))]
            files = [f for f in items if os.path.isfile(os.path.join(path, f))]
            
            # Show directories
            for i, d in enumerate(dirs[:10]):  # Limit to first 10
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                print(f"{prefix}{'└── ' if is_last_dir else '├── '}{d}/")
                
                # Check for DICOM files in this directory
                dir_path = os.path.join(path, d)
                dicom_count = len([f for f in os.listdir(dir_path) if f.endswith('.dcm')] if os.path.isdir(dir_path) else [])
                if dicom_count > 0:
                    print(f"{prefix}{'    ' if is_last_dir else '│   '}  ({dicom_count} DICOM files)")
                
                # Recurse
                new_prefix = prefix + ("    " if is_last_dir else "│   ")
                print_tree(dir_path, new_prefix, depth + 1)
            
            if len(dirs) > 10:
                print(f"{prefix}├── ... ({len(dirs) - 10} more directories)")
            
            # Show DICOM files
            dicom_files = [f for f in files if f.endswith('.dcm')]
            if dicom_files:
                print(f"{prefix}└── {len(dicom_files)} DICOM files")
            elif files and depth < max_depth - 1:
                # Show other files if no DICOM
                for i, f in enumerate(files[:5]):
                    print(f"{prefix}└── {f}")
                if len(files) > 5:
                    print(f"{prefix}    ... ({len(files) - 5} more files)")
                    
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
        except Exception as e:
            print(f"{prefix}[Error: {e}]")
    
    if os.path.exists(IPMN_PATH):
        print_tree(IPMN_PATH)
    else:
        print(f"❌ Path does not exist: {IPMN_PATH}")
    
    print("=" * 80)


def scan_ipmn_data():
    """Scan the IPMN directory structure to find all patients"""
    patients = []
    
    print(f"Scanning IPMN data directory: {IPMN_PATH}")
    
    if not os.path.exists(IPMN_PATH):
        print(f"❌ IPMN path does not exist: {IPMN_PATH}")
        # Try to explore parent directory
        parent_path = os.path.dirname(IPMN_PATH)
        if os.path.exists(parent_path):
            print(f"Parent directory exists: {parent_path}")
            print("Contents of parent directory:")
            for item in os.listdir(parent_path)[:20]:
                print(f"  - {item}")
        return patients
    
    # First, explore the structure to understand it better
    explore_ipmn_structure(max_depth=4)
    
    # Scan for batch folders
    batch_folders = [f for f in os.listdir(IPMN_PATH) if os.path.isdir(os.path.join(IPMN_PATH, f))]
    print(f"\nFound {len(batch_folders)} batch folders: {batch_folders[:5]}...")  # Show first 5
    
    for batch in batch_folders:
        batch_path = os.path.join(IPMN_PATH, batch)
        
        # Look for patient folders within each batch
        try:
            patient_folders = [f for f in os.listdir(batch_path) if os.path.isdir(os.path.join(batch_path, f))]
            print(f"  Batch {batch}: {len(patient_folders)} folders")
            
            for patient_folder in patient_folders:
                patient_path = os.path.join(batch_path, patient_folder)
                
                # Try different methods to extract MRN
                mrn = None
                
                # Method 1: Look for MRN pattern in folder name
                if 'MRN' in patient_folder:
                    # Try to extract MRN after "MRN" text
                    import re
                    mrn_match = re.search(r'MRN[_\s]*(\d+)', patient_folder)
                    if mrn_match:
                        mrn = mrn_match.group(1)
                
                # Method 2: If folder name is just numbers
                if not mrn and patient_folder.isdigit():
                    mrn = patient_folder
                
                # Method 3: Extract first sequence of digits
                if not mrn:
                    import re
                    digit_match = re.search(r'(\d{4,})', patient_folder)  # At least 4 digits
                    if digit_match:
                        mrn = digit_match.group(1)
                
                # Method 4: Use the whole folder name as ID
                if not mrn:
                    mrn = patient_folder
                
                # Debug: Show first few patient folders
                if len(patients) < 3:
                    print(f"    Patient folder: {patient_folder} -> MRN: {mrn}")
                
                # Check which modalities are available
                modalities = []
                modality_found = False
                
                # Check for modality folders at different levels
                # Level 1: Direct modality folders
                for modality in ["Art", "NonCon", "Ven", "Arterial", "Non-Contrast", "Venous"]:
                    modality_path = os.path.join(patient_path, modality)
                    if os.path.exists(modality_path):
                        # Check for Image subfolder
                        image_path = os.path.join(modality_path, "Image")
                        if os.path.exists(image_path):
                            # Check if there are DICOM files
                            files = [f for f in os.listdir(image_path) if f.endswith('.dcm')]
                            if files:
                                # Normalize modality name
                                if modality in ["Arterial", "Art"]:
                                    modalities.append("Art")
                                elif modality in ["Non-Contrast", "NonCon"]:
                                    modalities.append("NonCon")
                                elif modality in ["Venous", "Ven"]:
                                    modalities.append("Ven")
                                modality_found = True
                        else:
                            # Check if DICOM files are directly in modality folder
                            files = [f for f in os.listdir(modality_path) if f.endswith('.dcm')]
                            if files:
                                if modality in ["Arterial", "Art"]:
                                    modalities.append("Art")
                                elif modality in ["Non-Contrast", "NonCon"]:
                                    modalities.append("NonCon")
                                elif modality in ["Venous", "Ven"]:
                                    modalities.append("Ven")
                                modality_found = True
                
                # Level 2: Check for any subfolder that might contain DICOM files
                if not modality_found:
                    try:
                        subfolders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]
                        for subfolder in subfolders[:5]:  # Check first 5 subfolders
                            subfolder_path = os.path.join(patient_path, subfolder)
                            # Check for DICOM files
                            files = []
                            for root, dirs, filenames in os.walk(subfolder_path):
                                dicom_files = [f for f in filenames if f.endswith('.dcm')]
                                if dicom_files:
                                    files.extend(dicom_files)
                                    if len(patients) < 3:  # Debug for first few
                                        print(f"      Found {len(dicom_files)} DICOM files in {root}")
                                    break
                            
                            if files:
                                # Try to determine modality from folder name
                                subfolder_lower = subfolder.lower()
                                if any(x in subfolder_lower for x in ['art', 'arterial']):
                                    modalities.append("Art")
                                elif any(x in subfolder_lower for x in ['non', 'noncon']):
                                    modalities.append("NonCon")
                                elif any(x in subfolder_lower for x in ['ven', 'venous']):
                                    modalities.append("Ven")
                                else:
                                    # Default to NonCon if can't determine
                                    modalities.append("NonCon")
                                modality_found = True
                    except Exception as e:
                        print(f"      Error scanning {patient_path}: {e}")
                
                # Remove duplicates
                modalities = list(set(modalities))
                
                if modalities:
                    patients.append({
                        "MRN": mrn,
                        "BatchFolder": batch,
                        "PatientFolder": patient_folder,
                        "PatientPath": patient_path,
                        "Modalities": modalities
                    })
                    
                    if len(patients) <= 3:  # Show details for first few patients
                        print(f"      ✓ Added patient: MRN={mrn}, Modalities={modalities}")
                        
        except Exception as e:
            print(f"  Error processing batch {batch}: {e}")
    
    print(f"\nFound {len(patients)} patients with imaging data")
    if patients:
        print(f"Sample MRNs: {[p['MRN'] for p in patients[:10]]}")
    
    return patients


def get_dicom_folder(folder_path):
    """Get all DICOM files from a folder, including subdirectories"""
    dicom_paths = []
    
    # First try direct folder
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.dcm'):
                dicom_paths.append(os.path.join(folder_path, file))
    
    # If no DICOM files found, search subdirectories
    if not dicom_paths:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_paths.append(os.path.join(root, file))
            # Only go one level deep to avoid too many files
            if dicom_paths:
                break
    
    return sorted(dicom_paths)


def get_patients_with_clinical_data():
    """Get list of MRNs/Patient IDs from clinical CSV files"""
    print("Scanning clinical CSV files for patient IDs...")
    
    # Load clinical data
    clin_df = pd.read_csv(CLIN_CSV)
    
    # Get unique MRNs from clinical data
    clinical_mrns = set()
    patient_ids = set()
    
    if 'MRN' in clin_df.columns:
        mrns = clin_df['MRN'].dropna().astype(str).tolist()
        clinical_mrns.update(mrns)
        print(f"Found {len(mrns)} MRNs in clinical data")
        print(f"Sample MRNs: {mrns[:10]}")
    
    if 'PATIENT_ID' in clin_df.columns:
        pids = clin_df['PATIENT_ID'].dropna().astype(str).tolist()
        patient_ids.update(pids)
        print(f"Found {len(pids)} PATIENT_IDs in clinical data")
        print(f"Sample PATIENT_IDs: {pids[:10]}")
    
    # Combine both sets
    all_ids = clinical_mrns | patient_ids
    print(f"Total unique patient identifiers: {len(all_ids)}")
    
    return all_ids


def filter_imaging_patients_by_clinical(clinical_ids):
    """Filter imaging patients to only include those with clinical data"""
    print("Filtering imaging patients by clinical data availability...")
    
    # Check if we have a cached result
    cached_patients = load_checkpoint("filtered_patients.pkl")
    if cached_patients is not None:
        print(f"📁 Loaded {len(cached_patients)} filtered patients from checkpoint")
        return cached_patients
    
    # Get all imaging patients
    all_imaging_patients = scan_ipmn_data()
    print(f"Total imaging patients found: {len(all_imaging_patients)}")
    
    if not all_imaging_patients:
        return []
    
    # Show sample imaging MRNs for debugging
    imaging_mrns = [str(p['MRN']) for p in all_imaging_patients[:10]]
    print(f"Sample imaging MRNs: {imaging_mrns}")
    
    # Convert clinical IDs to set of strings for comparison
    clinical_ids_str = {str(id) for id in clinical_ids}
    
    # Filter to only patients with clinical data
    filtered_patients = []
    unmatched_mrns = []
    
    for patient in all_imaging_patients:
        patient_mrn = str(patient['MRN'])
        
        # Try exact match
        if patient_mrn in clinical_ids_str:
            filtered_patients.append(patient)
        else:
            # Try partial matches or different formats
            matched = False
            
            # Remove leading zeros and try again
            patient_mrn_no_zeros = patient_mrn.lstrip('0')
            if patient_mrn_no_zeros in clinical_ids_str:
                filtered_patients.append(patient)
                matched = True
            
            # Check if any clinical ID contains this MRN or vice versa
            if not matched:
                for cid in clinical_ids_str:
                    if patient_mrn in cid or cid in patient_mrn:
                        filtered_patients.append(patient)
                        matched = True
                        break
                    # Also try without leading zeros
                    if patient_mrn_no_zeros and (patient_mrn_no_zeros in cid or cid.lstrip('0') == patient_mrn_no_zeros):
                        filtered_patients.append(patient)
                        matched = True
                        break
            
            if not matched:
                unmatched_mrns.append(patient_mrn)
    
    print(f"Patients with both clinical and imaging data: {len(filtered_patients)}")
    print(f"Unmatched imaging MRNs: {len(unmatched_mrns)}")
    
    if filtered_patients:
        print(f"Sample matched patient MRNs: {[p['MRN'] for p in filtered_patients[:10]]}")
        
        # Save checkpoint
        save_checkpoint(filtered_patients, "filtered_patients.pkl")
        save_progress_log("filter_patients", {"total_patients": len(filtered_patients)})
    else:
        print("⚠️  No patients found with both clinical and imaging data!")
        print(f"Sample unmatched imaging MRNs: {unmatched_mrns[:10]}")
        
        # Try to find potential matches by showing both lists
        print("\nDEBUG: Comparing ID formats...")
        print(f"Clinical ID samples: {list(clinical_ids_str)[:10]}")
        print(f"Imaging MRN samples: {[str(p['MRN']) for p in all_imaging_patients[:10]]}")
        
        # Return all imaging patients for now to allow processing
        print("\n⚠️  WARNING: No ID matches found. Returning all imaging patients for processing.")
        print("   The final merge may fail, but embeddings will be generated.")
        filtered_patients = all_imaging_patients
        save_checkpoint(filtered_patients, "filtered_patients.pkl")
        save_progress_log("filter_patients", {"total_patients": len(filtered_patients), "warning": "no_matches"})
    
    return filtered_patients


def load_clinical_text_data():
    """Load and prepare clinical text data for patients"""
    print("Loading clinical text data...")
    
    # Load radiology and pathology reports
    rad_df = pd.read_csv(RAD_CSV)
    pat_df = pd.read_csv(PAT_CSV)
    
    # Create dictionaries for quick lookup
    radiology_texts = {}
    pathology_texts = {}
    
    # Process radiology reports
    if 'PATIENT_ID' in rad_df.columns and 'DATA_TEXT' in rad_df.columns:
        for _, row in rad_df.iterrows():
            patient_id = str(row['PATIENT_ID'])
            text = row['DATA_TEXT'] if pd.notna(row['DATA_TEXT']) else ""
            
            if patient_id in radiology_texts:
                radiology_texts[patient_id] += " " + text
            else:
                radiology_texts[patient_id] = text
    
    # Process pathology reports
    if 'PATIENT_ID' in pat_df.columns and 'DATA_TEXT' in pat_df.columns:
        for _, row in pat_df.iterrows():
            patient_id = str(row['PATIENT_ID'])
            text = row['DATA_TEXT'] if pd.notna(row['DATA_TEXT']) else ""
            
            if patient_id in pathology_texts:
                pathology_texts[patient_id] += " " + text
            else:
                pathology_texts[patient_id] = text
    
    print(f"Loaded radiology reports for {len(radiology_texts)} patients")
    print(f"Loaded pathology reports for {len(pathology_texts)} patients")
    
    return radiology_texts, pathology_texts


def process_patient_multimodal_ct_fusion(patient_info, available_modalities, preprocessor_args, clinical_data, radiology_texts, pathology_texts, encoder):
    """Process all available modalities for a patient and fuse into multimodal CT embeddings"""
    try:
        mrn = patient_info["MRN"]
        patient_id = str(mrn)
        
        # Get clinical data for this patient
        patient_clinical = clinical_data[clinical_data['PATIENT_ID'].astype(str) == patient_id]
        if patient_clinical.empty:
            patient_clinical = clinical_data[clinical_data['MRN'].astype(str) == patient_id] if 'MRN' in clinical_data.columns else pd.DataFrame()
        
        patient_row = patient_clinical.iloc[0] if not patient_clinical.empty else {}
        
        # Get text data
        radiology_text = radiology_texts.get(patient_id, "")
        pathology_text = pathology_texts.get(patient_id, "")
        
        # Create clinical text prompt
        clinical_text_prompt = create_clinical_text_prompt(patient_row, radiology_text, pathology_text)
        
        print(f"Processing {mrn} - multimodal CT fusion from {available_modalities}")
        
        # Create preprocessor
        preprocessor = DICOMPreprocessor(**preprocessor_args)
        
        # Collect embeddings from all available modalities
        modality_embeddings = {}
        modality_info = []
        
        for modality in available_modalities:
            # Find the DICOM files for this modality
            dicom_paths = []
            
            # Try standard paths first
            modality_path = os.path.join(patient_info["PatientPath"], modality)
            image_path = os.path.join(modality_path, "Image")
            
            if os.path.exists(image_path):
                dicom_paths = get_dicom_folder(image_path)
            elif os.path.exists(modality_path):
                dicom_paths = get_dicom_folder(modality_path)
            
            # If still no files, search the patient folder for any DICOM files
            if not dicom_paths:
                # Look for any folder containing the modality name
                for root, dirs, files in os.walk(patient_info["PatientPath"]):
                    folder_name = os.path.basename(root).lower()
                    modality_lower = modality.lower()
                    
                    # Check if folder name contains modality indication
                    if (modality_lower in folder_name or 
                        (modality == "Art" and any(x in folder_name for x in ["art", "arterial"])) or
                        (modality == "NonCon" and any(x in folder_name for x in ["non", "noncon"])) or
                        (modality == "Ven" and any(x in folder_name for x in ["ven", "venous"]))):
                        
                        temp_paths = get_dicom_folder(root)
                        if temp_paths:
                            dicom_paths = temp_paths
                            break
            
            if not dicom_paths:
                print(f"  No DICOM files found for {mrn} - {modality}")
                continue
            
            print(f"  Processing {len(dicom_paths)} DICOM files for {mrn} - {modality}")
            
            # Process DICOM files
            processed_slices = []
            for i, dicom_path in enumerate(dicom_paths):
                if i > 100:  # Limit number of slices to avoid memory issues
                    break
                processed_image = preprocessor.preprocess(dicom_path)
                if processed_image is not None:
                    processed_slices.append(processed_image)
            
            if processed_slices:
                # Sample slices if too many
                if len(processed_slices) > 50:
                    # Take every nth slice to get approximately 50 slices
                    step = len(processed_slices) // 50
                    processed_slices = processed_slices[::step]
                
                # Encode with MedGemma using clinical context
                multimodal_embeddings = encoder.encode_multimodal(processed_slices, clinical_text_prompt)
                
                if multimodal_embeddings.size > 0:
                    modality_embeddings[modality] = multimodal_embeddings
                    modality_info.append({
                        'modality': modality,
                        'slice_count': len(processed_slices),
                        'embedding_shape': multimodal_embeddings.shape
                    })
                    
                    print(f"  ✅ {modality}: {multimodal_embeddings.shape} multimodal embedding extracted")
        
        # Store embeddings for each modality and create fused representation
        if modality_embeddings:
            result = {
                "MRN": mrn,
                "BatchFolder": patient_info["BatchFolder"],
                "clinical_text_used": clinical_text_prompt,
                "has_radiology_report": bool(radiology_text),
                "has_pathology_report": bool(pathology_text),
            }
            
            # Store individual modality embeddings
            for modality, embeddings in modality_embeddings.items():
                result[f"multimodal_{modality}_embeddings"] = embeddings.astype(np.float32).tobytes()
                result[f"multimodal_{modality}_embedding_shape"] = embeddings.shape
            
            # Create fused embedding by concatenating all modalities
            all_embeddings = list(modality_embeddings.values())
            if len(all_embeddings) > 1:
                # Concatenate all slices from all modalities
                fused_embedding = np.concatenate(all_embeddings, axis=0)
            else:
                fused_embedding = all_embeddings[0]
            
            result["multimodal_ct_fused_embeddings"] = fused_embedding.astype(np.float32).tobytes()
            result["multimodal_ct_fused_embedding_shape"] = fused_embedding.shape
            result["modalities_used"] = ",".join(modality_embeddings.keys())
            result["modality_count"] = len(modality_embeddings)
            
            print(f"✅ Multimodal CT fusion complete for {mrn}: {fused_embedding.shape} from {len(modality_embeddings)} modalities")
            
            # Memory cleanup
            del modality_embeddings, fused_embedding
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
        
        print(f"❌ No valid multimodal embeddings extracted for {mrn}")
        return None
        
    except Exception as e:
        print(f"Error processing {patient_info.get('MRN', 'unknown')} multimodal CT fusion: {str(e)}")
        traceback.print_exc()
        return None


def process_multimodal_embeddings_chunked(filtered_patients, clinical_data, radiology_texts, pathology_texts):
    """Process multimodal embeddings in chunks with checkpointing"""
    print("Processing multimodal embeddings in chunks...")
    
    # Check for existing results
    existing_results_path = os.path.join(CHECKPOINT_DIR, "multimodal_results_accumulated.parquet")
    processed_patients_path = os.path.join(CHECKPOINT_DIR, "processed_patients.pkl")
    
    if os.path.exists(existing_results_path) and os.path.exists(processed_patients_path):
        print("📁 Loading existing multimodal results...")
        accumulated_results = pd.read_parquet(existing_results_path)
        processed_patients = load_checkpoint("processed_patients.pkl")
        print(f"Found {len(accumulated_results)} existing results for {len(processed_patients)} patients")
    else:
        accumulated_results = pd.DataFrame()
        processed_patients = set()
    
    # Create tasks for unprocessed patients
    all_tasks = []
    for patient in filtered_patients:
        patient_mrn = str(patient['MRN'])
        if patient_mrn not in processed_patients:
            available_modalities = [mod for mod in patient["Modalities"] if mod in ["Art", "NonCon", "Ven"]]
            if available_modalities:
                all_tasks.append((patient, available_modalities))
    
    if not all_tasks:
        print("✅ All patients already processed!")
        return accumulated_results if not accumulated_results.empty else pd.DataFrame()
    
    print(f"Processing {len(all_tasks)} remaining patients in chunks of {CHUNK_SIZE}")
    
    preprocessor_args = {
        "target_size": 224,
        "normalize": True,
        "to_three_channels": True,
    }
    
    # Initialize encoder once
    encoder = MedGemmaMultimodalEncoder(device_id=0 if torch.cuda.is_available() else None)
    
    # Process in chunks
    processed_in_session = set()
    
    for chunk_idx in range(0, len(all_tasks), CHUNK_SIZE):
        chunk_tasks = all_tasks[chunk_idx:chunk_idx + CHUNK_SIZE]
        chunk_num = chunk_idx // CHUNK_SIZE + 1
        total_chunks = (len(all_tasks) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        print(f"\n🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_tasks)} patients)...")
        
        # Process chunk
        chunk_results_list = []
        for patient, available_modalities in tqdm(chunk_tasks, desc=f"Chunk {chunk_num}"):
            try:
                result = process_patient_multimodal_ct_fusion(
                    patient, available_modalities, preprocessor_args, 
                    clinical_data, radiology_texts, pathology_texts, encoder
                )
                if result:
                    chunk_results_list.append(result)
                    processed_in_session.add(str(patient['MRN']))
                
                # Clear GPU cache periodically
                if len(chunk_results_list) % 3 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"❌ Error processing {patient['MRN']}: {e}")
                continue
        
        # Add chunk results to accumulated results
        if chunk_results_list:
            chunk_df = pd.DataFrame(chunk_results_list)
            
            if accumulated_results.empty:
                accumulated_results = chunk_df.copy()
            else:
                # Before concatenating, check for duplicates
                existing_mrns = set(accumulated_results['MRN'].astype(str))
                new_results = []
                duplicate_count = 0
                
                for _, row in chunk_df.iterrows():
                    if str(row['MRN']) not in existing_mrns:
                        new_results.append(row)
                    else:
                        duplicate_count += 1
                        print(f"  ⚠️  Skipping duplicate MRN: {row['MRN']}")
                
                if duplicate_count > 0:
                    print(f"  Skipped {duplicate_count} duplicate MRNs")
                
                if new_results:
                    new_df = pd.DataFrame(new_results)
                    accumulated_results = pd.concat([accumulated_results, new_df], ignore_index=True)
            
            print(f"✅ Chunk {chunk_num} complete: {len(chunk_results_list)} new results")
            print(f"📊 Total accumulated results: {len(accumulated_results)}")
        
        # Save progress every few chunks
        if chunk_num % SAVE_EVERY == 0 or chunk_num == total_chunks:
            print(f"💾 Saving progress after chunk {chunk_num}...")
            
            # Update processed patients list
            processed_patients.update(processed_in_session)
            
            # Save checkpoints
            save_checkpoint(accumulated_results, "multimodal_results_accumulated.parquet")
            save_checkpoint(processed_patients, "processed_patients.pkl")
            save_progress_log("multimodal_chunk", {
                "chunk": chunk_num, 
                "total_chunks": total_chunks,
                "results_count": len(accumulated_results),
                "processed_patients": len(processed_patients)
            })
            
            print(f"✅ Progress saved! Processed {len(processed_patients)} patients so far")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)  # Brief pause between chunks
    
    print(f"\n🎉 Multimodal processing complete! Total results: {len(accumulated_results)}")
    
    # Final deduplication check
    if not accumulated_results.empty:
        initial_count = len(accumulated_results)
        accumulated_results = accumulated_results.drop_duplicates(subset=['MRN'], keep='first')
        final_count = len(accumulated_results)
        if initial_count != final_count:
            print(f"⚠️  Removed {initial_count - final_count} duplicate MRNs from final results")
            print(f"📊 Final unique patient count: {final_count}")
        
        save_checkpoint(accumulated_results, "multimodal_embeddings_final.parquet")
        save_progress_log("multimodal_embeddings", {"patients": len(accumulated_results)})
    
    return accumulated_results


def compute_survival_times(df):
    """Compute survival times with appropriate study cutoff date"""
    print("Computing survival times...")
    
    # Convert date columns to datetime
    date_columns = ['surgery_date', 'vital_status_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Determine study cutoff
    if 'vital_status_date' in df.columns:
        study_cutoff = df['vital_status_date'].max()
    else:
        study_cutoff = pd.Timestamp('2023-12-31')
    
    print(f"Using study cutoff date: {study_cutoff}")
    
    # Initialize survival columns
    df['survival_time_days'] = np.nan
    df['survival_time_years'] = np.nan
    df['event_observed'] = 0
    df['has_survival_data'] = False
    
    # Compute survival times
    if 'surgery_date' in df.columns and 'vital_status' in df.columns:
        # For patients who died
        dead_mask = (df['vital_status'].str.upper() == 'DEAD') & df['surgery_date'].notna()
        if 'vital_status_date' in df.columns:
            dead_mask = dead_mask & df['vital_status_date'].notna()
            df.loc[dead_mask, 'survival_time_days'] = (df.loc[dead_mask, 'vital_status_date'] - df.loc[dead_mask, 'surgery_date']).dt.days
            df.loc[dead_mask, 'event_observed'] = 1
            df.loc[dead_mask, 'has_survival_data'] = True
        
        # For patients who are alive (censored)
        alive_mask = (df['vital_status'].str.upper() == 'ALIVE') & df['surgery_date'].notna()
        df.loc[alive_mask, 'survival_time_days'] = (study_cutoff - df.loc[alive_mask, 'surgery_date']).dt.days
        df.loc[alive_mask, 'event_observed'] = 0
        df.loc[alive_mask, 'has_survival_data'] = True
    
    # Convert days to years
    df.loc[df['survival_time_days'].notna(), 'survival_time_years'] = df.loc[df['survival_time_days'].notna(), 'survival_time_days'] / 365.25
    
    # Create binary vital status
    df['vital_status_binary'] = (df['vital_status'].str.upper() == 'DEAD').astype(int)
    
    # Summary
    with_survival = df['has_survival_data'].sum()
    events = df['event_observed'].sum()
    print(f"Patients with survival data: {with_survival}")
    print(f"Events observed (deaths): {events}")
    
    return df


def create_final_dataset():
    """Combine all data into final dataset"""
    print("Creating final MedGemma multimodal dataset...")
    
    # Check for cached final dataset
    cached_final = load_checkpoint("final_medgemma_dataset.parquet")
    if cached_final is not None:
        print("📁 Using cached final dataset")
        return cached_final
    
    # Load clinical data
    clin_df = pd.read_csv(CLIN_CSV)
    print(f"Clinical data shape: {clin_df.shape}")
    print(f"Clinical data columns: {clin_df.columns.tolist()}")
    
    # Check for duplicate columns in clinical data
    duplicate_cols = clin_df.columns[clin_df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"⚠️  Found duplicate columns in clinical data: {duplicate_cols}")
        # Keep only the first occurrence of duplicate columns
        clin_df = clin_df.loc[:, ~clin_df.columns.duplicated()]
        print(f"Clinical data shape after removing duplicates: {clin_df.shape}")
    
    # Get patients with clinical data
    clinical_mrns = get_patients_with_clinical_data()
    
    # Filter imaging patients
    filtered_patients = filter_imaging_patients_by_clinical(clinical_mrns)
    
    if not filtered_patients:
        print("❌ No patients found with both clinical and imaging data")
        return pd.DataFrame()
    
    # Load clinical text data
    radiology_texts, pathology_texts = load_clinical_text_data()
    
    # Process multimodal embeddings
    multimodal_results = process_multimodal_embeddings_chunked(
        filtered_patients, clin_df, radiology_texts, pathology_texts
    )
    
    print(f"Multimodal results shape: {multimodal_results.shape}")
    
    # Check for duplicate MRNs in multimodal results
    if not multimodal_results.empty:
        mrn_counts = multimodal_results['MRN'].value_counts()
        duplicates = mrn_counts[mrn_counts > 1]
        if len(duplicates) > 0:
            print(f"⚠️  Found {len(duplicates)} duplicate MRNs in multimodal results:")
            print(duplicates.head())
            print("Removing duplicates by keeping the first occurrence...")
            multimodal_results = multimodal_results.drop_duplicates(subset=['MRN'], keep='first')
            print(f"After deduplication: {multimodal_results.shape}")
    
    # Start with clinical data
    final_df = clin_df.copy()
    
    # Merge multimodal embeddings
    if not multimodal_results.empty:
        # Debug: Show columns in both dataframes
        print(f"Clinical data columns: {final_df.columns.tolist()[:10]}...")
        print(f"Multimodal results columns: {multimodal_results.columns.tolist()}")
        
        # Prepare for merging
        final_df['PATIENT_ID'] = final_df['PATIENT_ID'].astype(str)
        multimodal_results['PATIENT_ID'] = multimodal_results['MRN'].astype(str)
        
        # Try merging on PATIENT_ID
        overlap = set(final_df['PATIENT_ID']) & set(multimodal_results['PATIENT_ID'])
        print(f"Patient ID overlap: {len(overlap)} patients")
        
        if len(overlap) > 0:
            # Drop MRN column from multimodal_results to avoid conflicts
            multimodal_cols = [col for col in multimodal_results.columns if col != 'MRN']
            final_df = final_df.merge(multimodal_results[multimodal_cols], on="PATIENT_ID", how="left")
            print(f"✅ Successfully merged multimodal embeddings for {len(overlap)} patients")
        else:
            # Try MRN-based merge
            if 'MRN' in final_df.columns:
                final_df['MRN'] = final_df['MRN'].astype(str)
                multimodal_results['MRN_imaging'] = multimodal_results['MRN'].astype(str)
                
                # Check if MRN column already exists in multimodal_results
                if 'MRN' in multimodal_results.columns:
                    # Drop the original MRN column to avoid duplication
                    multimodal_results = multimodal_results.drop(columns=['MRN'])
                
                # Rename PATIENT_ID to avoid conflicts and merge on MRN
                multimodal_results = multimodal_results.drop(columns=['PATIENT_ID'])
                multimodal_results = multimodal_results.rename(columns={"MRN_imaging": "MRN"})
                
                # Show overlap info
                mrn_overlap = set(final_df['MRN']) & set(multimodal_results['MRN'])
                print(f"MRN overlap: {len(mrn_overlap)} patients")
                if len(mrn_overlap) > 0:
                    print(f"Sample overlapping MRNs: {list(mrn_overlap)[:5]}")
                
                final_df = final_df.merge(multimodal_results, on="MRN", how="left")
                print(f"✅ Successfully merged via MRN for {len(mrn_overlap)} patients")
            else:
                print("❌ No ID overlap found between clinical and multimodal data!")
                
                # Save multimodal embeddings separately
                multimodal_output_path = os.path.join(OUTPUT_DIR, "multimodal_embeddings_standalone.parquet")
                multimodal_results.to_parquet(multimodal_output_path, index=False)
                print(f"💾 Saved standalone multimodal embeddings to: {multimodal_output_path}")
                
                # Show ID format comparison
                print("\nID Format Analysis:")
                print(f"Clinical PATIENT_IDs: {list(final_df['PATIENT_ID'].astype(str).unique())[:5]}")
                if 'MRN' in final_df.columns:
                    print(f"Clinical MRNs: {list(final_df['MRN'].astype(str).unique())[:5]}")
                print(f"Multimodal MRNs: {list(multimodal_results['MRN'].astype(str).unique())[:5]}")
                
                # Add empty multimodal columns to maintain structure
                multimodal_cols = [col for col in multimodal_results.columns if col not in ['MRN', 'BatchFolder']]
                for col in multimodal_cols:
                    final_df[col] = None
                
                print("⚠️  Created placeholder columns for multimodal data in final dataset")
    
    # Compute survival times
    final_df = compute_survival_times(final_df)
    
    # Save final dataset
    save_checkpoint(final_df, "final_medgemma_dataset.parquet")
    save_progress_log("final_dataset", {"total_patients": len(final_df)})
    
    return final_df


def clear_checkpoints():
    """Clear all checkpoint files"""
    if os.path.exists(CHECKPOINT_DIR):
        files_to_remove = [
            "filtered_patients.pkl",
            "multimodal_results_accumulated.parquet",
            "processed_patients.pkl",
            "multimodal_embeddings_final.parquet",
            "final_medgemma_dataset.parquet"
        ]
        
        for file in files_to_remove:
            file_path = os.path.join(CHECKPOINT_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Removed checkpoint: {file}")
        
        print("✅ Checkpoints cleared")


def main():
    """Main function to generate the final MedGemma multimodal dataset"""
    print("Starting IPMN MedGemma multimodal processing pipeline...")
    print(f"📁 Checkpoints will be saved to: {CHECKPOINT_DIR}")
    
    # Option to clear checkpoints (uncomment if needed)
    # clear_checkpoints()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    try:
        # Generate final dataset
        final_df = create_final_dataset()
        
        if final_df.empty:
            print("❌ No data to save")
            return
        
        # Save main dataset
        output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
        final_df.to_parquet(output_file_path, index=False)
        
        print(f"\n🎉 Final MedGemma multimodal dataset saved to: {output_file_path}")
        print(f"Dataset shape: {final_df.shape}")
        
        # Summary statistics
        print(f"\nData Summary:")
        print(f"- Total patients: {len(final_df)}")
        print(f"- Patients with survival data: {final_df['has_survival_data'].sum()}")
        
        # Check multimodal embeddings
        multimodal_cols = [col for col in final_df.columns if 'multimodal_' in col and 'embeddings' in col and 'shape' not in col]
        has_merged_multimodal = False
        for col in multimodal_cols:
            count = final_df[col].notna().sum()
            if count > 0:
                has_merged_multimodal = True
            print(f"- Patients with {col}: {count}")
        
        # If no merged multimodal data, check standalone file
        if not has_merged_multimodal:
            standalone_path = os.path.join(OUTPUT_DIR, "multimodal_embeddings_standalone.parquet")
            if os.path.exists(standalone_path):
                standalone_df = pd.read_parquet(standalone_path)
                print(f"\n📊 Standalone Multimodal Embeddings Summary:")
                print(f"- Total patients with multimodal embeddings: {len(standalone_df)}")
                
                # Check each modality
                for modality in ['Art', 'NonCon', 'Ven']:
                    col = f'multimodal_{modality}_embeddings'
                    if col in standalone_df.columns:
                        count = standalone_df[col].notna().sum()
                        print(f"- Patients with {modality} embeddings: {count}")
                
                # Check fused embeddings
                if 'multimodal_ct_fused_embeddings' in standalone_df.columns:
                    count = standalone_df['multimodal_ct_fused_embeddings'].notna().sum()
                    print(f"- Patients with fused CT embeddings: {count}")
                
                print(f"\n⚠️  Note: Multimodal embeddings were generated but could not be merged with clinical data due to ID mismatch.")
                print(f"   Please check the ID formats in both datasets for proper merging.")
        
        # Save summary
        summary_info = {
            "dataset_path": output_file_path,
            "total_patients": len(final_df),
            "patients_with_survival": int(final_df['has_survival_data'].sum()),
            "multimodal_embedding_counts": {}
        }
        
        # Count embeddings in main dataset
        for col in multimodal_cols:
            if col in final_df.columns:
                summary_info["multimodal_embedding_counts"][col] = int(final_df[col].notna().sum())
        
        # Add standalone embeddings info if they exist
        standalone_path = os.path.join(OUTPUT_DIR, "multimodal_embeddings_standalone.parquet")
        if os.path.exists(standalone_path):
            standalone_df = pd.read_parquet(standalone_path)
            summary_info["standalone_embeddings"] = {
                "path": standalone_path,
                "total_patients": len(standalone_df),
                "embedding_counts": {}
            }
            
            standalone_cols = [col for col in standalone_df.columns if 'multimodal_' in col and 'embeddings' in col and 'shape' not in col]
            for col in standalone_cols:
                summary_info["standalone_embeddings"]["embedding_counts"][col] = int(standalone_df[col].notna().sum())
        
        summary_path = os.path.join(OUTPUT_DIR, "IPMN_MedGemma_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        print(f"📋 Dataset summary saved to: {summary_path}")
        
        # Also save to EAGLE directory
        eagle_output_path = "/proj/rasool_lab_projects/Aakash/EAGLE/data/IPMN_MedGemma_Multimodal.parquet"
        os.makedirs(os.path.dirname(eagle_output_path), exist_ok=True)
        final_df.to_parquet(eagle_output_path)
        print(f"📁 Also saved to EAGLE directory: {eagle_output_path}")
        
        print("\n✅ IPMN MedGemma multimodal processing complete!")
        
    except Exception as e:
        print(f"\n❌ Error in main pipeline: {e}")
        traceback.print_exc()
        save_progress_log("error", {"error_message": str(e), "success": False})
        raise


if __name__ == "__main__":
    main()