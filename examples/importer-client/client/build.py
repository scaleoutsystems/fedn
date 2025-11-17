import os
from scaleoututil.helpers.plugins.numpyhelper import Helper
import numpy as np



def build():
    output_dir = os.environ.get("SCALEOUT_BUILD_OUTPUT_DIR", ".")
    np.random.seed(42)
    params = np.random.rand(10).astype(np.float32)
    Helper().save([params], os.path.join(output_dir, "seed.npz"))
    print(f"Created seed.npz with 10 random parameters.")

if __name__ == "__main__":
    build()
