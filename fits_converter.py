from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

fname = "sdss_data/images/GALAXY_369.fits"
hdu_list = fits.open(fname)
image_data = hdu_list[0].data

# Scale between 1st and 99th percentile
vmin, vmax = np.percentile(image_data, (1, 99))
print(f"Displaying image with vmin={vmin}, vmax={vmax}")

plt.imshow(image_data, cmap='gray', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.show()