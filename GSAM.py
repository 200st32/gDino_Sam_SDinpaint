import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
from PIL import ImageFilter
from transformers import SamModel, SamProcessor
import numpy as np
import os
from tqdm import tqdm

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("./myoutput/gdino_test_val100.png")
    #plt.show()

def gdino(device, model, processor, image):

    #model_id = "IDEA-Research/grounding-dino-base"
    #device = "cuda"

    #processor = AutoProcessor.from_pretrained(model_id)
    #model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    #image_path = "/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/img/val100.png"
    #image = Image.open(image_path)
    #image = image.filter(ImageFilter.SHARPEN)
    #image = image.filter(ImageFilter.SHARPEN)
    # Check for cats and remote controls
    text = "cars. trucks. vehicle"

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    results = results[0]
    #print(results)
    #plot_results(image, results['scores'].tolist(), results['labels'], results['boxes'].tolist())

    return results['boxes'].tolist()

def show_mask(mask, random_color=False):
    #if random_color:
        #color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    #else:
    color = np.array([225/255, 225/255, 225/255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #ax.imshow(mask_image)
    return mask_image

def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    #fig, axes = plt.subplots(1, 1, figsize=(15, 15))

    black_img = np.zeros_like(np.array(raw_image))
    #axes.imshow(black_img)
    #axes.axis("off")
    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      r = show_mask(mask)
      black_img += r
      #axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
    #plt.show()
    #plt.savefig("./myoutput/sam_test_val100.png")
    mask_image = Image.fromarray((black_img).astype(np.uint8))
    mask_image.save("./myoutput/sam_test_mask_val100.png")

def sam(bboxes, device, model, processor, image):

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    #processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    #image_path = "/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/img/val100.png"
    #image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    input_boxes = []
    input_boxes.append(bboxes)

    inputs = processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)

    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores
    #print("=========")
    #print(masks[0])
    #show_masks_on_image(image, masks[0], scores)
    
    masks = masks[0]

    if isinstance(masks, torch.Tensor):
        masks = list(masks)
    combined_mask_tensor = torch.stack(masks, dim=0).any(dim=0).to(torch.uint8)
    combined_mask_np = (combined_mask_tensor.numpy() * 255).astype(np.uint8)
    combined_mask_np = np.squeeze(combined_mask_np)
    combined_mask_image = Image.fromarray(combined_mask_np)
    #combined_mask_image.save("./myoutput/sam_test_mask_val100.png")
    return combined_mask_image

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)
    s_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")

    folder = "/home/cap6411.student1/CVsystem/assignment/hw8/cityscapes/val/img/"
    for filename in tqdm(os.listdir(folder)):
        image = Image.open(os.path.join(folder,filename))
        width, height = image.size
        
        bboxes = gdino(device, gd_model, gd_processor, image)
        
        if len(bboxes) > 0:
            #mymask = sam(bboxes, device, s_model, s_processor, image)
            mymask = Image.new("L", (width, height), 0)  # "L" mode for grayscale
            draw = ImageDraw.Draw(mymask)
            for (x_min, y_min, x_max, y_max) in bboxes:
                draw.rectangle([x_min, y_min, x_max, y_max], fill=255)
        else:
            black_img = np.zeros_like(np.array(image))
            mymask = Image.fromarray(black_img)
        #mymask.save(f"./mymask/mask_{filename}")
        mymask.save(f"./bb_mask/mask_{filename}")
        



