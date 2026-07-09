import os

def build_library(source_file, library_dir, chunk_size_mb=5, max_volumes=5):
    """
    Slices a massive XML file into smaller 'volumes' for the AGI to visually read.
    Limits to max_volumes for the initial run to prove the concept and prevent filling disk space instantly.
    """
    if not os.path.exists(source_file):
        print(f"Error: Could not find {source_file}")
        return
        
    os.makedirs(library_dir, exist_ok=True)
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    print(f"Chunking {source_file} into {chunk_size_mb} MB volumes...")
    
    with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
        volume = 1
        while volume <= max_volumes:
            chunk = f.read(chunk_size_bytes // 4) # approx characters
            if not chunk:
                break
                
            vol_path = os.path.join(library_dir, f"Volume_{volume}.xml")
            with open(vol_path, 'w', encoding='utf-8') as out_f:
                out_f.write(chunk)
                
            print(f"Created {vol_path} ({chunk_size_mb} MB)")
            volume += 1
            
    print("\nLibrary construction complete (initial batch).")

if __name__ == "__main__":
    source = r"C:\Users\User\Downloads\TRAIN\enwiktionary-latest-pages-meta-current.xml"
    lib_dir = r"C:\Users\User\Downloads\TRAIN\Library"
    build_library(source, lib_dir, chunk_size_mb=2, max_volumes=5)
