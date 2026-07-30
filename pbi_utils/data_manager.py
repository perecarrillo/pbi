from colorama import Fore
import numpy as np
import pandas as pd
import subprocess
import torch
import h5py
import os
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, List
from pbi_utils.logging import Logging

logger = Logging()


class EmbeddingsManager(ABC):
    """
    Abstract class for embeddings storage managers. For now, only H5py is implemented, but other storage backends can be added by extending this class.
    """

    @abstractmethod
    def save_embedding(
        self, id: int, embedding: torch.Tensor, model_name: str, overwrite: bool = False
    ) -> None:
        """
        Save a single embedding.

        :param id: The identifier for the embedding.
        :type id: int
        :param embedding: The embedding tensor to be saved.
        :type embedding: torch.Tensor
        :param model_name: The name of the model associated with the embedding. Used to locate the correct storage.
        :type model_name: str
        :param overwrite: Whether to overwrite an existing embedding with the same id. Default is False.
        :type overwrite: bool
        """
        pass

    @abstractmethod
    def load_embedding(
        self, id: int, model_name: str, remove: bool = False, device: str = "cpu", keep_shape: bool = False
    ) -> torch.Tensor | None:
        """
        Load a single embedding.

        :param id: The identifier for the embedding.
        :type id: int
        :param model_name: The name of the model associated with the embedding. Used to locate the correct storage.
        :type model_name: str
        :param remove: Whether to remove the embedding from storage after loading it. Default is False.
        :type remove: bool
        :param device: The device to load the embedding onto. Default is "cpu".
        :type device: str
        :param keep_shape: Whether to preserve original tensor shape (2D for raw chunks). Default is False.
        :type keep_shape: bool
        :return: The loaded embedding tensor, or None if not found.
        :rtype: Tensor | None
        """
        pass

    @abstractmethod
    def save_embeddings_batch(
        self,
        ids: List[int],
        embeddings: List[torch.Tensor],
        model_name: str,
        overwrite: bool = False,
        silent: bool = False,
    ) -> None:
        """
        Save a batch of embeddings.

        :param ids: The list of identifiers for the embeddings.
        :type ids: List[int]
        :param embeddings: The list of embedding tensors to be saved.
        :type embeddings: List[torch.Tensor]
        :param model_name: The name of the model associated with the embeddings. Used to locate the correct storage.
        :type model_name: str
        :param overwrite: Whether to overwrite existing embeddings with the same ids. Default is False.
        :type overwrite: bool
        :param silent: Whether to suppress progress output. Default is False.
        :type silent: bool
        """
        pass

    @abstractmethod
    def load_embedding_batch(
        self,
        ids: List[int],
        model_name: str,
        remove: bool = False,
        device: str = "cpu",
        silent: bool = False,
        keep_shape: bool = False,
    ) -> List[torch.Tensor]:
        """
        Load a batch of embeddings.

        :param ids: The list of identifiers for the embeddings.
        :type ids: List[int]
        :param model_name: The name of the model associated with the embeddings. Used to locate the correct storage.
        :type model_name: str
        :param remove: Whether to remove the embeddings from storage after loading them. Default is False.
        :type remove: bool
        :param device: The device to load the embeddings onto. Default is "cpu".
        :type device: str
        :param silent: Whether to suppress progress output. Default is False.
        :type silent: bool
        :return: The list of loaded embedding tensors in the device specified.
        :rtype: List[Tensor]
        """
        pass

    @abstractmethod
    def remove_key(
        self, id: int | str, model_name: str, ignore_not_found: bool
    ) -> None:
        """
        Remove a specific embedding from storage.

        :param id: The identifier for the embedding.
        :type id: int | str
        :param model_name: The name of the model associated with the embedding. Used to locate the correct storage.
        :type model_name: str
        :param ignore_not_found: Whether to ignore the case when the embedding is not found.
        :type ignore_not_found: bool
        """
        pass

    @abstractmethod
    def has_key(self, id: int, model_name: str) -> bool:
        """
        Check if a specific embedding exists in storage.

        :param id: The identifier for the embedding.
        :type id: int
        :param model_name: The name of the model associated with the embedding. Used to locate the correct storage.
        :type model_name: str
        :return: True if the embedding exists, False otherwise.
        :rtype: bool
        """
        pass


class H5pyEmbeddingsManager(EmbeddingsManager):
    """
    Embeddings manager that uses H5py files for storage. Each model's embeddings are stored in a separate H5 file within the specified base path.
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        logger.info(f"Embeddings will be stored or read from {base_path}")

    def save_embedding(
        self, id: int, embedding: torch.Tensor, model_name: str, overwrite: bool = False
    ):
        id = str(id)  # type: ignore

        # ensure CPU + numpy
        data = embedding.detach().cpu().float().numpy()

        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            if id in f:
                if not overwrite:
                    logger.debug(
                        f"{id} already exists, skipping it. To overwrite the value, use overwrite=True"
                    )
                    return
                del f[id]
            f.create_dataset(id, data=data, compression="gzip")

    def save_embeddings_batch(
        self,
        ids: List[int],
        embeddings: List[torch.Tensor],
        model_name: str,
        overwrite: bool = False,
        silent: bool = False,
    ):
        emb_shape = embeddings[0].shape
        eq_shape = list(map(lambda x: x.shape == emb_shape, embeddings))
        all_eq_shape = np.all(eq_shape)

        logger.debug(
            f"Saving {len(ids)} embeddings for model {model_name} to {self.base_path}. "
            + f"Shape: {emb_shape}"
            if all_eq_shape
            else f"Found different shapes at position 0 ({emb_shape}) and {eq_shape.index(False)} ({embeddings[eq_shape.index(False)].shape})"
        )
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            for id, embedding in tqdm(
                zip(ids, embeddings),
                total=len(ids),
                desc="Saving embeddings",
                disable=silent,
            ):
                id = str(id)  # type: ignore
                # ensure CPU + numpy
                data = embedding.detach().cpu().float().numpy()
                if not overwrite and id in f:
                    logger.debug(
                        f"{id} already exists, skipping it. To overwrite the value, use overwrite=True"
                    )
                else:
                    self.remove_key(id, model_name, ignore_not_found=True)
                    f.create_dataset(id, data=data, compression="gzip")

    def load_embedding(
        self, id: int, model_name: str, remove: bool = False, device: str = "cpu", keep_shape: bool = False
    ) -> torch.Tensor | None:
        id = str(id)  # type: ignore
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "r") as f:
            if id not in f:
                logger.warning(f"{id} not found when loading embedding")
                embed = None
            else:
                raw_data = torch.from_numpy(f[id][:]).to(device=device)
                embed = raw_data if keep_shape else raw_data.flatten()
                if remove:
                    self.remove_key(id, model_name)

        return embed

    def load_embedding_batch(
        self,
        ids: List[int],
        model_name: str,
        remove: bool = False,
        device: str = "cpu",
        silent: bool = False,
        keep_shape: bool = False,
    ) -> List[torch.Tensor]:
        logger.debug(
            f"Loading {len(ids)} embeddings for model {model_name} from {self.base_path}"
        )
        result = []
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "r") as f:
            for id in tqdm(ids, desc="Loading embeddings", disable=silent):
                id = str(id)  # type: ignore
                if id not in f:
                    logger.warning(f"{id} not found when loading batch")
                else:
                    raw_data = torch.from_numpy(f[id][:]).to(device=device)
                    result.append(raw_data if keep_shape else raw_data.flatten())
                    if remove:
                        self.remove_key(id, model_name)
        return result

    def remove_key(
        self, id: int | str, model_name: str, ignore_not_found: bool = False
    ):
        id = str(id)  # type: ignore
        with h5py.File(os.path.join(self.base_path, model_name + ".h5"), "a") as f:
            if id in f:
                del f[id]
            elif not ignore_not_found:
                logger.info(f"{id} not found when trying to remove it.")

    def has_key(self, id: int, model_name: str) -> bool:
        filepath = os.path.join(self.base_path, model_name + ".h5")

        if not os.path.isfile(filepath):
            return False

        id = str(id)  # type: ignore
        with h5py.File(filepath, "r") as f:
            return id in f


class InputManager(ABC):
    """
    Abstract class for data input managers.

    For now, only the Perphect format is implemented, but other formats can be added by extending this class.
    """

    @abstractmethod
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the data and return bacteria, phages and couples dataframes.

        :return: A tuple containing bacteria, phages and couples dataframes, with the following formats:
            - Bacteria DataFrame: columns=['bacterium_id', 'bacterium_sequence']
            - Phages DataFrame: columns=['phage_id', 'phage_sequence']
            - Couples DataFrame: columns=['bacterium_id', 'phage_id', 'interaction_type']
        :rtype: Tuple[DataFrame, DataFrame, DataFrame]
        """
        pass


class PerphectDataInput(InputManager):
    """
    Input manager for Perphect data format.
    """

    def __init__(self, input_paths) -> None:
        self.bacteria_path = input_paths.bacteria_df
        self.phages_path = input_paths.phages_df
        self.couples_path = input_paths.couples_df

        if not os.path.isfile(self.bacteria_path):
            logger.warning(f"Bacteria file not found. Path tried: {self.bacteria_path}")
        if not os.path.isfile(self.phages_path):
            logger.warning(f"Phages file not found. Path tried: {self.phages_path}")
        if not os.path.isfile(self.couples_path):
            logger.warning(f"Couples file not found. Path tried: {self.couples_path}")

        logger.info(
            f"Perphect input files will be read from {self.bacteria_path}, {self.phages_path} and {self.couples_path}"
        )

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        # wc = int(subprocess.run([f"cat {self.base_path}*.csv | wc -l"], capture_output=True, shell=True).stdout) # line count of the files, but read_csv does not reach the end idk why

        logger.info("Reading csv files...")
        couples_df = pd.read_csv(self.couples_path)
        bacteria_df = pd.read_csv(self.bacteria_path)
        phages_df = pd.read_csv(self.phages_path)
        return bacteria_df, phages_df, couples_df
