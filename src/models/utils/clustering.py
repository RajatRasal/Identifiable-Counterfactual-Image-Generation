import torch
from sklearn import preprocessing
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.utils.wrapper import SlotWrapper


@torch.no_grad
def knn(slot_wrapper: SlotWrapper, dl: DataLoader, device: torch.device):
    # get slots
    all_slots = []
    all_attns = []
    for i, image in enumerate(tqdm(dl, desc="Slots")):
        image = image.to(device)
        slots, attns = slot_wrapper.slot_mask(image)
        all_slots.append(slots.cpu())
        all_attns.append(attns.cpu())
    all_slots = torch.cat(all_slots, dim=0)
    all_attns = torch.cat(all_attns, dim=0)

    # cluster slots
    k_means = KMeans(n_clusters=slot_wrapper.n_slots)
    classes = k_means.fit_predict(preprocessing.normalize(all_slots.numpy()))

    # choose new slots for composition
    comp_slots = []
    attns = {}
    for i in range(slot_wrapper.n_slots):
        mask_i = classes == i
        slots_i = all_slots[mask_i]
        attns_i = all_attns[mask_i]
        shuffled_idxs = torch.randint(
            low=0,
            high=slots_i.shape[0],
            size=(len(dl) * dl.batch_size,),
        )
        shuffled_slots_i = slots_i[shuffled_idxs]
        comp_slots.append(shuffled_slots_i)
        attns[i] = attns_i[shuffled_idxs]

    # compose new images
    comp_slots = torch.stack(comp_slots).transpose(0, 1)
    comp_slots = DataLoader(
        TensorDataset(comp_slots),
        batch_size=dl.batch_size,
        num_workers=dl.num_workers,
        pin_memory=True,
    )
    comps = [
        slot_wrapper.compose(comp_slot[0].cuda()).cpu()
        for comp_slot in tqdm(comp_slots, desc="Comps")
    ]
    comps = torch.cat(comps, dim=0)

    return comps, attns
