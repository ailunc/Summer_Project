import argparse, os, torch
import numpy as np
from monai.networks.nets import UNet
from torchvision import transforms as T
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--size", type=int, nargs=2, default=[512,512])
    ap.add_argument("--rgb", action="store_true", help="treat input as RGB")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_ch = 3 if args.rgb else 1
    model = UNet(spatial_dims=2, in_channels=in_ch, out_channels=args.num_classes,
                 channels=(32,64,128,256), strides=(2,2,2), dropout=0.1).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd); model.eval()

    img = Image.open(args.image).convert("RGB" if args.rgb else "L")
    img = T.Resize(args.size, interpolation=T.InterpolationMode.BILINEAR)(img)
    ten = T.ToTensor()(img)
    ten = T.Normalize(mean=[0.5]*in_ch, std=[0.5]*in_ch)(ten)
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = model(ten[None].to(device))
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)

    from PIL import Image as PILImage
    out = PILImage.fromarray(pred, mode="P")
    palette = []
    colors = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,128),(128,128,0),(0,128,128)]
    for i in range(256):
        c = colors[i] if i < len(colors) else (i, i, i)
        palette.extend(c)
    out.putpalette(palette[:768])
    os.makedirs(os.path.dirname(args.out_mask), exist_ok=True)
    out.save(args.out_mask)
    print("Saved:", args.out_mask)

if __name__ == "__main__":
    main()
