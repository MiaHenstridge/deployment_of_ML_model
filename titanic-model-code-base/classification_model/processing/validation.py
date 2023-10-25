# classification_model/processing/validation.py


from typing import List, Optional, Tuple, Union

import numpy as np 
import pandas as pd 
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config
from classification_model.processing.data_manager import pre_pipeline_preparation





def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
	"""Check model inputs for unprocessale values."""

	pre_processed = pre_pipeline_preparation(dataframe=input_data)
	validated_data = pre_processed[config.model_config.features].copy()
	errors = None

	try:
		# replace numpy nans so that pydantic can validate
		MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
	except ValidationError as error:
		errors = error.json()

	return validated_data, errors




class TitanicDataInputSchema(BaseModel):
	"""Define a schema for each row of input data"""
	pclass: Optional[int]
	name: Optional[str]
	sex: Optional[str]
	age: Optional[int]
	sibsp: Optional[int]
	parch: Optional[int]
	ticket: Optional[int]
	fare: Optional[float]
	cabin: Optional[str]
	embarked: Optional[str]
	boat: Optional[Union[str, int]]
	body: Optional[int]


class MultipleTitanicDataInputs(BaseModel):
	"""Define a schema for multiple rows of input data"""
	inputs: List[TitanicDataInputSchema]