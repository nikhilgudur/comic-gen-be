from PIL import Image, ImageDraw, ImageFont
import io
import base64
from typing import Tuple, Optional, List


def add_caption_to_image(
    image: Image.Image,
    caption: str,
    position: str = 'bottom',
    font_size: int = 20,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Add a text caption to an image"""
    # Create a copy of the image
    img_with_caption = image.copy()
    width, height = img_with_caption.size

    # Create a drawing context
    draw = ImageDraw.Draw(img_with_caption)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text size
    text_width, text_height = draw.textbbox((0, 0), caption, font=font)[2:4]

    # Create the caption background
    if position == 'bottom':
        caption_box = Image.new(
            'RGB', (width, text_height + 2 * padding), bg_color)
        img_with_caption.paste(
            caption_box, (0, height - text_height - 2 * padding))
        text_position = ((width - text_width) // 2,
                         height - text_height - padding)
    elif position == 'top':
        caption_box = Image.new(
            'RGB', (width, text_height + 2 * padding), bg_color)
        img_with_caption.paste(caption_box, (0, 0))
        text_position = ((width - text_width) // 2, padding)

    # Draw the text
    draw = ImageDraw.Draw(img_with_caption)
    draw.text(text_position, caption, font=font, fill=text_color)

    return img_with_caption


def create_comic_strip(
    images: List[Image.Image],
    captions: Optional[List[str]] = None,
    panel_spacing: int = 10,
    border_width: int = 2,
    border_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Combine multiple images into a comic strip with borders"""
    if not images:
        raise ValueError("No images provided")

    # Ensure all images have the same height
    base_height = images[0].height
    resized_images = []

    for img in images:
        aspect = img.width / img.height
        new_width = int(base_height * aspect)
        resized_images.append(img.resize(
            (new_width, base_height), Image.LANCZOS))

    # Calculate the total width
    total_width = sum(img.width for img in resized_images) + \
        panel_spacing * (len(images) - 1)

    # Create the comic strip canvas
    comic = Image.new('RGB', (total_width, base_height), (255, 255, 255))

    # Paste each image with a border
    x_offset = 0
    for i, img in enumerate(resized_images):
        # If captions are provided, add them to the panels
        if captions and i < len(captions):
            img = add_caption_to_image(img, captions[i])

        # Add a border by drawing on the comic
        draw = ImageDraw.Draw(comic)
        draw.rectangle(
            [(x_offset, 0), (x_offset + img.width - 1, base_height - 1)],
            outline=border_color,
            width=border_width
        )

        # Paste the image
        comic.paste(img, (x_offset, 0))
        x_offset += img.width + panel_spacing

    return comic
