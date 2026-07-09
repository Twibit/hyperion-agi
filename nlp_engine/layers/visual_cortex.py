import numpy as np

class VisualCortex:
    """
    Simulates the V1/V2 Visual Cortex.
    Uses basic receptive fields (Gabor-like random sparse projections) to convert 
    raw 2D pixel arrays into ultra-sparse 1D orthongonal spikes for the Hippocampus.
    """
    def __init__(self, output_dim, image_shape=(28, 28)):
        self.output_dim = output_dim
        self.image_shape = image_shape
        self.input_dim = image_shape[0] * image_shape[1]
        
        # Biological Gabor Filters (V1 simple cells)
        self.receptive_fields = self._generate_gabor_filters()
        
    def _generate_gabor_filters(self):
        filters = np.zeros((self.input_dim, self.output_dim), dtype=np.float32)
        h, w = self.image_shape
        
        y, x = np.mgrid[0:h, 0:w]
        
        for i in range(self.output_dim):
            # Random biological parameters for each neuron's receptive field
            theta = np.random.uniform(0, np.pi)       # Orientation
            sigma = np.random.uniform(2, 6)           # Gaussian envelope size
            lambd = np.random.uniform(3, 10)          # Spatial wavelength
            gamma = np.random.uniform(0.5, 1.0)       # Spatial aspect ratio
            psi = np.random.uniform(0, np.pi)         # Phase offset
            
            # Receptive field center
            x0 = np.random.uniform(0, w)
            y0 = np.random.uniform(0, h)
            
            # Gabor equation
            x_theta = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
            y_theta = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
            
            gb = np.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * np.cos(2 * np.pi * x_theta / lambd + psi)
            
            filters[:, i] = gb.flatten()
            
        return filters
        
    def encode_image(self, image_array):
        """Converts an image into a sparse spike array."""
        flat_image = image_array.flatten()
        
        # O(N) Matrix multiplication (Retina -> V1)
        # Only process active pixels to save FLOPs
        active_pixels = np.nonzero(flat_image > 0)[0]
        membrane = np.zeros(self.output_dim, dtype=np.float32)
        
        for p in active_pixels:
            membrane += (self.receptive_fields[p] * float(flat_image[p]))
            
        # K-Winner-Take-All (Lateral Inhibition in V1)
        # Force exact orthogonality (sparsity)
        k_winners = 15 # Only the 15 strongest visual features fire
        
        spikes = np.zeros(self.output_dim, dtype=bool)
        if np.any(membrane > 0):
            winners = np.argsort(membrane)[-k_winners:]
            spikes[winners] = True
            
        return spikes

    def imagine_image(self, spikes):
        """Generative Pathway (Top-Down): Converts sparse brain spikes back into a 2D image."""
        flat_image = np.dot(self.receptive_fields, spikes.astype(np.float32))
        
        # Normalize to 0-255 grayscale
        if np.max(flat_image) > 0:
            flat_image = (flat_image / np.max(flat_image)) * 255.0
            
        return flat_image.reshape(self.image_shape).astype(np.uint8)

    def imagine_3d(self, spikes, filepath="dream.obj"):
        """Generative 3D Pathway: Projects spikes into a 3D voxel grid and saves as OBJ."""
        voxel_dim = 16
        total_voxels = voxel_dim ** 3
        
        voxels = np.zeros(total_voxels, dtype=bool)
        
        for i, spike in enumerate(spikes):
            if spike:
                # Each visual feature projects to a unique 3D geometrical coordinate
                np.random.seed(i) # Deterministic projection for this specific feature
                active_indices = np.random.choice(total_voxels, size=20, replace=False)
                voxels[active_indices] = True
                
        # Reshape to 3D grid
        grid = voxels.reshape((voxel_dim, voxel_dim, voxel_dim))
        
        # Write to .obj file (simple voxel cubes)
        with open(filepath, 'w') as f:
            f.write("# Hyperion 3D Visual Imagination (Mind's Eye)\\n")
            vertex_offset = 1
            for x in range(voxel_dim):
                for y in range(voxel_dim):
                    for z in range(voxel_dim):
                        if grid[x, y, z]:
                            f.write(f"v {x} {y} {z}\\n")
                            f.write(f"v {x+1} {y} {z}\\n")
                            f.write(f"v {x+1} {y+1} {z}\\n")
                            f.write(f"v {x} {y+1} {z}\\n")
                            f.write(f"v {x} {y} {z+1}\\n")
                            f.write(f"v {x+1} {y} {z+1}\\n")
                            f.write(f"v {x+1} {y+1} {z+1}\\n")
                            f.write(f"v {x} {y+1} {z+1}\\n")
                            
                            f.write(f"f {vertex_offset} {vertex_offset+1} {vertex_offset+2} {vertex_offset+3}\\n") 
                            f.write(f"f {vertex_offset+4} {vertex_offset+5} {vertex_offset+6} {vertex_offset+7}\\n") 
                            f.write(f"f {vertex_offset} {vertex_offset+1} {vertex_offset+5} {vertex_offset+4}\\n") 
                            f.write(f"f {vertex_offset+3} {vertex_offset+2} {vertex_offset+6} {vertex_offset+7}\\n") 
                            f.write(f"f {vertex_offset} {vertex_offset+3} {vertex_offset+7} {vertex_offset+4}\\n") 
                            f.write(f"f {vertex_offset+1} {vertex_offset+2} {vertex_offset+6} {vertex_offset+5}\\n") 
                            
                            vertex_offset += 8
        print(f"      [Visual Cortex (3D)] Rendered 3D thought to {filepath}")
