"""Datasets: add a ``.d`` once by path, refer to it by id afterwards."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..app import AppState, get_state, not_found
from ..schemas import DatasetAdd, DatasetOut, DatasetPatch

router = APIRouter(tags=["datasets"])


@router.post("/datasets", response_model=DatasetOut, status_code=201)
def add_dataset(body: DatasetAdd, state: AppState = Depends(get_state)):
    ds = state.registry.add(body.path, body.name, body.group)
    out = state.registry.describe(ds)
    state.bus.publish("dataset", out)
    return out


@router.get("/datasets", response_model=list[DatasetOut])
def list_datasets(state: AppState = Depends(get_state)):
    return [state.registry.describe(ds) for ds in state.registry.list()]


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: str, state: AppState = Depends(get_state)):
    ds = state.registry.get(dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    return state.registry.describe(ds)


@router.patch("/datasets/{dataset_id}", response_model=DatasetOut)
def patch_dataset(dataset_id: str, body: DatasetPatch, state: AppState = Depends(get_state)):
    # Only fields the client actually sent; "group": null clears the group.
    changes = body.model_dump(include=body.model_fields_set)
    if "name" in changes and changes["name"] is None:
        del changes["name"]
    ds = state.registry.update(dataset_id, **changes)
    if ds is None:
        raise not_found("dataset", dataset_id)
    out = state.registry.describe(ds)
    state.bus.publish("dataset", out)
    return out


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, purge_cache: bool = False,
                   state: AppState = Depends(get_state)):
    ds = state.registry.get(dataset_id)
    if ds is None:
        raise not_found("dataset", dataset_id)
    if purge_cache:
        entry = state.registry.cache_entry(ds)
        if entry is not None:
            state.cache.remove(entry.key)
    state.registry.remove(dataset_id)
    state.bus.publish("dataset_removed", {"id": dataset_id})
