import math
import logging
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pystac_client
import stackstac
import torch
import yaml
from box import Box
from rasterio.enums import Resampling
from shapely.geometry import Point
from torchvision.transforms import v2

from src.model import ClayMAEModule


STAC_API = os.getenv("STAC_API", "https://earth-search.aws.element84.com/v1")
COLLECTION = os.getenv("COLLECTION", "sentinel-2-l2a")
COLLECTIONS = [COLLECTION]
METADATA_PATH = os.getenv("METADATA_PATH", "./clay-model/configs/metadata.yaml")
MODEL_CKPT = "https://clay-model-ckpt.s3.amazonaws.com/v0.5.7/mae_v0.5.7_epoch-13_val-loss-0.3098.ckpt"
CLOUD_COVERAGE_LIMIT = 20
ASSETS = [
    "blue",
    "green",
    "red",
    "nir",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "swir16",
    "swir32",
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)


def get_model_from_ckpt(ckpt):
    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path=METADATA_PATH, shuffle=False, mask_ratio=0
    )
    model.eval()
    model = model.to(device)
    return model


def get_catalog_items(lat, lon, start, end, collections=COLLECTIONS):
    # Search the catalogue
    bb_offset = 1e-5
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=collections,
        datetime=f"{start}/{end}",
        bbox=(lon - bb_offset, lat - bb_offset, lon + bb_offset, lat + bb_offset),
        max_items=100,
        query={
            "eo:cloud_cover": {"lt": 20},
            # "sentinel:valid_cloud_cover": {"eq": True}
        },
        sortby="properties.eo:cloud_cover",
    )
    all_items = search.item_collection()
    if not all_items:
        raise ValueError(f"Could not find any data for given bounding box {lat} {lon}")

    # Reduce to one per date (there might be some duplicates
    # based on the location)
    items, dates = [], []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())
            cloud_coverage = item.properties.get("eo:cloud_cover", -1)
            if cloud_coverage > CLOUD_COVERAGE_LIMIT:
                logging.warning(
                    f"Warning: cloud coverage is {cloud_coverage}>{CLOUD_COVERAGE_LIMIT} for input {lat} {lon}"
                )

    return items


def get_bounds(lat, lon, size, gsd, epsg):
    # Convert point of interest into the image projection
    # (assumes all images are in the same projection)
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]

    # Create bounds in projection
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    return bounds


# Prep datetimes embedding using a normalization function from the model code.
def normalize_timestamp(date):
    week = date.isocalendar().week * 2 * np.pi / 52
    hour = date.hour * 2 * np.pi / 24

    return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))


# Prep lat/lon embedding using the
def normalize_latlon(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180

    return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))


def get_datacube(
    lat, lon, items, gsd=10, size=256, assets=ASSETS, collection=COLLECTION
):
    # Extract coordinate system from first item
    epsg = items[0].properties["proj:epsg"]
    # gsds = [items[0].assets[asset].extra_fields['gsd'] for asset in assets]
    bounds = get_bounds(lat, lon, gsd, size, epsg)

    # Retrieve the pixel values, for the bounding box in
    # the target projection. In this example we use only
    # the RGB and NIR bands.
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float32",
        rescale=False,
        fill_value=0,
        assets=assets,
        resampling=Resampling.nearest,
    )
    stack = stack.compute()

    # Extract mean, std, and wavelengths from metadata
    platform = collection
    metadata = Box(yaml.safe_load(open(METADATA_PATH)))
    mean, std, waves = [], [], []
    # Use the band names to get the correct values in the correct order.
    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    # Prepare the normalization transform function using the mean and std values.
    transform = v2.Compose(
        [
            v2.Normalize(mean=mean, std=std),
        ]
    )

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    # Prepare additional information
    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor([gsd], device=device),  # todo get gsd from items?
        "waves": torch.tensor(waves, device=device),
    }

    return datacube, stack


MODEL = get_model_from_ckpt(MODEL_CKPT)


def get_embeddings(
    lat,
    lon,
    start_date,
    end_date,
    size=256,
    collection=COLLECTION,
    assets=ASSETS,
    model=None,
):
    if not model:
        model = MODEL

    items = get_catalog_items(lat, lon, start_date, end_date)
    datacube, stack = get_datacube(lat, lon, items, size=size, assets=assets)

    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)
    # todo understand that this is correct to return, is it THE single embedding
    # confirm with bruno about per platform issue
    return unmsk_patch[:, 0, :].cpu().numpy(), stack


def get_embeddings_many(latlons, start_date, end_date, size=256, model=None):
    return [
        get_embeddings(lat, lon, start_date, end_date, size=size)[0]
        for lat, lon in latlons
    ]
