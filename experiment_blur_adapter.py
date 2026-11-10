"""Owned experiment extension: source RGB GaussianBlur before spatial preprocessing."""
from PIL import ImageFilter
from physical_probe import apply_source_space_intervention as canonical_intervention

def apply_source_space_intervention(image, intervention, level):
    if intervention == 'blur' and any(abs(float(level)-x)<1e-12 for x in (.55,.6)):
        return image.convert('RGB').filter(ImageFilter.GaussianBlur(radius=float(level)))
    return canonical_intervention(image, intervention, level)
