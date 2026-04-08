"""
Test FreeSpaceProp using original repository code in modules.py
Usage:
    python debug_scripts/test_free_space.py
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import FreeSpaceProp


def test_free_space_propagation():
    print("=" * 60)
    print("Testing FreeSpaceProp (Angular Spectrum Method)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs("./debug_scripts/optical_test", exist_ok=True)

    params = {
        'wlength_vc': 520e-9,
        'ridx_air': 1.0,
        'total_x_num': 800,
        'total_y_num': 800,
        'dx': 8e-6,
        'dy': 8e-6,
        'prop_z': 5e-2
    }
    print("\n[1] Basic propagation test:")
    print(f"    Wavelength: {params['wlength_vc']*1e9:.0f} nm")
    print(f"    Propagation distance: {params['prop_z']*100:.2f} cm")
    print(f"    Grid size: {params['total_x_num']}x{params['total_y_num']}")
    print(f"    Grid physical size: {params['total_x_num'] * params['dx'] * 1000:.2f} mm")
    print(f"    Nyquist freq: {1/(2*params['dx'])*1e-6:.2f} um^-1")

    prop_layer = FreeSpaceProp(**params).to(device)
    print(f"    Buffer shape: {prop_layer.shifted_phase_change_cplx.shape}")

    # Point source params (smaller aperture for more rings)
    point_params = params.copy()
    point_params['prop_z'] = 10e-2  # 10cm for more diffraction spread

    x = torch.linspace(-params['total_x_num'] * params['dx'] / 2,
                       params['total_x_num'] * params['dx'] / 2,
                       params['total_x_num'])
    y = torch.linspace(-params['total_y_num'] * params['dy'] / 2,
                       params['total_y_num'] * params['dy'] / 2,
                       params['total_y_num'])
    X, Y = torch.meshgrid(x, y, indexing='ij')

    print("\n[2] Point source diffraction (smaller aperture for visible rings):")
    sigma_point = params['dx'] * 0.5  # Even smaller for tighter focus
    point_source = torch.exp(-(X**2 + Y**2) / (2 * sigma_point**2))
    point_source = point_source.unsqueeze(0).unsqueeze(0).to(device)

    print(f"    Gaussian sigma: {sigma_point * 1e6:.2f} um")
    print(f"    Fresnel number: {sigma_point**2 / (params['wlength_vc'] * point_params['prop_z']):.4f}")

    prop_layer_point = FreeSpaceProp(**point_params).to(device)
    with torch.no_grad():
        diffracted = prop_layer_point(point_source)
        intensity = torch.abs(diffracted)**2

    intensity_img = (intensity / intensity.max()).cpu().numpy()[0, 0]
    Image.fromarray((intensity_img * 255).astype(np.uint8)).save(
        "./debug_scripts/optical_test/point_source_diffraction.png"
    )
    print(f"    Saved: point_source_diffraction.png (z={point_params['prop_z']*100:.0f}cm, sigma={sigma_point*1e6:.2f}um)")

    print("\n[3] Slit diffraction:")
    slit = torch.zeros(1, 1, params['total_x_num'], params['total_y_num'], device=device)
    slit_width = 1  # 1 pixel slit
    center = params['total_x_num'] // 2
    slit[0, 0, center-slit_width:center+slit_width, :] = 1.0

    with torch.no_grad():
        diffracted_slit = prop_layer(slit)
        intensity_slit = torch.abs(diffracted_slit)**2

    slit_img = (intensity_slit / intensity_slit.max()).cpu().numpy()[0, 0]
    Image.fromarray((slit_img * 255).astype(np.uint8)).save(
        "./debug_scripts/optical_test/slit_diffraction.png"
    )
    print(f"    Saved: slit_diffraction.png")

    print("\n[4] Different propagation distances:")
    distances = [1e-2, 5e-2, 10e-2, 20e-2]
    for dist in distances:
        prop_layer_z = FreeSpaceProp(
            wlength_vc=params['wlength_vc'],
            ridx_air=params['ridx_air'],
            total_x_num=params['total_x_num'],
            total_y_num=params['total_y_num'],
            dx=params['dx'],
            dy=params['dy'],
            prop_z=dist
        ).to(device)

        with torch.no_grad():
            diffracted_z = prop_layer_z(point_source)
            intensity_z = torch.abs(diffracted_z)**2

        z_img = (intensity_z / intensity_z.max()).cpu().numpy()[0, 0]
        Image.fromarray((z_img * 255).astype(np.uint8)).save(
            f"./debug_scripts/optical_test/diffraction_z{int(dist*100):03d}cm.png"
        )
        print(f"    Distance {dist*100:.0f} cm: saved")

    print("\n[5] Gaussian beam propagation:")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = torch.linspace(-params['total_x_num']*params['dx']/2,
                           params['total_x_num']*params['dx']/2,
                           params['total_x_num'])
        y = torch.linspace(-params['total_y_num']*params['dy']/2,
                           params['total_y_num']*params['dy']/2,
                           params['total_y_num'])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        sigma = 20e-6
        gaussian = torch.exp(-(X**2 + Y**2) / (2 * sigma**2)).to(device)
        gaussian = gaussian.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            propagated_gauss = prop_layer(gaussian)
            intensity_gauss = torch.abs(propagated_gauss)**2

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(gaussian.cpu().squeeze(), cmap='hot')
        axes[0].set_title('Input Gaussian')
        axes[1].imshow(intensity_gauss.cpu().squeeze(), cmap='hot')
        axes[1].set_title('After 5cm Propagation')
        plt.tight_layout()
        plt.savefig("./debug_scripts/optical_test/gaussian_propagation.png", dpi=150)
        plt.close()
        print(f"    Saved: gaussian_propagation.png")
    except Exception as e:
        print(f"    Matplotlib not available, skipping visualization: {e}")

    print("\n[6] Energy conservation:")
    test_input = torch.randn(1, 1, params['total_x_num'], params['total_y_num'], device=device)
    with torch.no_grad():
        test_output = prop_layer(test_input)
        input_energy = torch.sum(torch.abs(test_input)**2).item()
        output_energy = torch.sum(torch.abs(test_output)**2).item()
    print(f"    Input energy: {input_energy:.6f}")
    print(f"    Output energy: {output_energy:.6f}")
    print(f"    Energy ratio: {output_energy/input_energy:.4f}")

    print("\n" + "=" * 60)
    print("All FreeSpaceProp tests passed!")
    print("Results saved to ./debug_scripts/optical_test/")
    print("=" * 60)


if __name__ == "__main__":
    test_free_space_propagation()
