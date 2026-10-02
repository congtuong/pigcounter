"""Training-only augmentation for a shaking outdoor pig camera."""
from __future__ import annotations


def outdoor_augmentations() -> list:
    import albumentations as A

    # Pixel-only effects keep boxes unchanged. YOLO handles geometry and boxes.
    return [
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 21), p=0.85),
            A.GaussianBlur(blur_limit=(3, 7), p=0.15),
        ], p=0.45),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.25, p=1),
            A.RandomGamma(gamma_limit=(65, 140), p=1),
        ], p=0.4),
        # One weather effect at a time to avoid routinely hiding the pigs.
        A.OneOf([
            A.RandomRain(rain_type='drizzle', drop_width=1, blur_value=3,
                         brightness_coefficient=0.85, p=1),
            A.RandomFog(fog_coef_range=(0.08, 0.3), alpha_coef=0.08, p=1),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=4,
                           shadow_intensity_range=(0.2, 0.45), p=1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=80,
                             num_flare_circles_range=(1, 3), p=0.5),
        ], p=0.25),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
        A.ImageCompression(quality_range=(55, 95), p=0.15),
    ]


def augmentation_settings(profile: str) -> dict:
    if profile == 'standard':
        return {}  # Use the library defaults.
    if profile != 'outdoor':
        raise ValueError(f'Unknown augmentation profile: {profile}')
    return dict(
        augmentations=outdoor_augmentations(),
        degrees=12.0, translate=0.15, scale=0.35, shear=2.0,
        perspective=0.0003, fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.35, hsv_v=0.25,
        mosaic=0.3, close_mosaic=15, mixup=0.0,
    )
