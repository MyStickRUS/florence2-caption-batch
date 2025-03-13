import torch
from pathlib import Path
from functools import partial
from typing import Iterator, List, Tuple
from PIL import Image
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_imports
import time
from tqdm import tqdm

# ====================== Добавляем функцию clean_caption из caption.py ======================

default_replacements = [
    ("the image features", ""),
    ("the image shows", ""),
    ("the image depicts", ""),
    ("the image is", ""),
    ("in this image", ""),
    ("in the image", ""),
    ("gigi hadid", "a woman"),
]

def clean_caption(cap: str, replacements=None) -> str:
    """
    Убирает некоторые фразы, юникод-символы, точку, дубликаты и т.д. 
    Делает результат строчным, убирает пустые элементы после запятой.
    """
    if replacements is None:
        replacements = default_replacements

    # Удаляем переносы строк
    cap = cap.replace("\n", ", ")
    cap = cap.replace("\r", ", ")

    # Меняем точки на запятые
    cap = cap.replace(".", ",")

    # Убираем кавычки
    cap = cap.replace("\"", "")

    # Удаляем не-ASCII символы (unicode)
    cap = cap.encode('ascii', 'ignore').decode('ascii')

    # Приводим к нижнему регистру
    cap = cap.lower()

    # Убираем повторяющиеся пробелы
    cap = " ".join(cap.split())

    # Заменяем заданные фразы/подстроки
    for old, new in replacements:
        # Если замена начинается с '*', удаляем всю строку при совпадении начала
        if old.startswith('*'):
            search_text = old[1:]  # убираем '*'
            if cap.startswith(search_text):
                cap = ""
        else:
            cap = cap.replace(old.lower(), new.lower())

    # Разделяем по запятым, обрезаем пробелы и убираем пустые куски
    parts = [x.strip() for x in cap.split(",")]
    parts = [x for x in parts if x]  # убираем пустые

    # Убираем дубли
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    parts = seen

    # Склеиваем обратно в строку
    return ", ".join(parts)

# ===========================================================================================

torch.set_float32_matmul_precision("high")

# Configuration options
OVERWRITE = True          # Разрешаем перезапись существующих .txt файлов
PREPEND_STRING = ""       # Строка, добавляемая перед основным текстом
APPEND_STRING = ""        # Строка, добавляемая после основного текста
BATCH_SIZE = 5            # Размер батча при генерации описаний
PRINT_PROCESSING_STATUS = False  # Печатать статус обработки
PRINT_CAPTIONS = False           # Печатать подписи в консоль
DETAIL_MODE = 2                 # Уровень детализации подписи (используется во Florence-2)

print(f"Captioning with batch size: {BATCH_SIZE}")

def fixed_get_imports(filename: str | Path) -> List[str]:
    imports = get_imports(filename)
    # Исключаем flash_attn для модели Florence2
    return [imp for imp in imports if imp != "flash_attn"] if str(filename).endswith("modeling_florence2.py") else imports

def download_and_load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device available: {device}')

    model_path = Path("models") / model_name.replace('/', '_')
    if not model_path.exists():
        print(f"Downloading {model_name} model to: {model_path}")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)

    print(f"Loading model {model_name}...")
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded.")

    model = torch.compile(model, mode="reduce-overhead")
    return model, processor

def load_image_paths_recursive(folder_path: str) -> Iterator[Path]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return (
        path for path in Path(folder_path).rglob("*")
        if path.suffix.lower() in valid_extensions and (OVERWRITE or not path.with_suffix('.txt').exists())
    )

def run_model_batch(
    image_paths: List[Path],
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    task: str = 'caption',
    num_beams: int = 3,
    max_new_tokens: int = 1024,
    detail_mode: int = DETAIL_MODE
) -> List[str]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Карточки prompt в зависимости от уровня детализации
    prompt = {
        1: '<CAPTION>',
        2: '<DETAILED_CAPTION>',
        3: '<MORE_DETAILED_CAPTION>'
    }.get(detail_mode, '<MORE_DETAILED_CAPTION>')

    inputs = {
        "input_ids": [],
        "pixel_values": []
    }

    for image_path in image_paths:
        if PRINT_PROCESSING_STATUS:
            print(f"Processing image: {image_path}")
        with Image.open(image_path).convert("RGB") as img:
            input_data = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
                do_rescale=False
            )
            inputs["input_ids"].append(input_data["input_ids"])
            inputs["pixel_values"].append(input_data["pixel_values"])

    # Собираем в тензоры
    inputs["input_ids"] = torch.cat(inputs["input_ids"]).to(device)
    inputs["pixel_values"] = torch.cat(inputs["pixel_values"]).to(device).to(torch.bfloat16)

    # Генерация
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=num_beams,
    )

    # Декодируем результаты
    results = processor.batch_decode(generated_ids, skip_special_tokens=False)

    # Удаляем спецтокены <s>, </s>, <pad>
    captions = []
    for result in results:
        captions.append(
            result.replace('</s>', '').replace('<s>', '').replace('<pad>', '')
        )
    return captions

def process_images_recursive(
    paths: Iterator[Path],
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    batch_size: int = 8
) -> Tuple[int, float]:
    start_time = time.time()
    total_images = 0

    # Превращаем итератор в список
    path_list = list(paths)
    num_batches = len(path_list) // batch_size + (1 if len(path_list) % batch_size > 0 else 0)

    for i in tqdm(range(num_batches), desc="Processing batches"):
        batch = path_list[i * batch_size : (i + 1) * batch_size]
        captions = run_model_batch(batch, model, processor, task='caption', detail_mode=DETAIL_MODE)

        for path_obj, raw_caption in zip(batch, captions):
            # Добавляем при необходимости префикс/суффикс
            combined_caption = f"{PREPEND_STRING}{raw_caption}{APPEND_STRING}"

            # =================== Важная часть: запускаем clean_caption ===================
            cleaned_caption = clean_caption(combined_caption)
            # ============================================================================

            if PRINT_CAPTIONS:
                print(f"Caption for {path_obj}: {cleaned_caption}")

            # Записываем итоговую подпись в txt-файл
            path_obj.with_suffix('.txt').write_text(cleaned_caption)
            total_images += 1

    total_time = time.time() - start_time
    return total_images, total_time

# ===================== Основной запуск =====================
model_name = 'microsoft/Florence-2-large'
model, processor = download_and_load_model(model_name)

folder_path = Path(__file__).parent / "input"
total_images, total_time = process_images_recursive(
    load_image_paths_recursive(folder_path),
    model,
    processor,
    batch_size=BATCH_SIZE
)

print(f"Total images captioned: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")

if total_images > 0:
    print(f"Average time per image: {total_time / total_images:.2f} seconds")
else:
    print("No images were processed, so no average time to display.")

file_count = len(list(folder_path.iterdir()))
print(f"Total files in folder: {file_count}")
