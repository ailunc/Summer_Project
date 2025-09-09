import argparse, os, torch, nibabel as nib
import numpy as np
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing,
    ScaleIntensityRange, EnsureType, SaveImage
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--pixdim", type=float, nargs=3, default=[1.5,1.5,2.0])
    ap.add_argument("--orientation", default="RAS")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(spatial_dims=3, in_channels=1, out_channels=args.num_classes,
                 channels=(32,64,128,256,512), strides=(2,2,2,2), dropout=0.1).to(device)
    sd = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes=args.orientation),
        Spacing(pixdim=args.pixdim, mode="bilinear"),
        ScaleIntensityRange(a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureType()
    ])

    img = transforms(args.image)
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = model(torch.from_numpy(img[None]).float().to(device))
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().astype(np.uint16)[0]

    # Save prediction NIfTI (copy affine from input if possible)
    nii = nib.load(args.image)
    out = nib.Nifti1Image(pred, nii.affine, nii.header)
    nib.save(out, args.out_pred)
    print("Saved:", args.out_pred)

if __name__ == "__main__":
    main()
