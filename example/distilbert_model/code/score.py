import os
import json
import logging
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def init():
    """    
    This function is called when the container is initialized/started, typically after create/update of the deployment.    
    You can write the logic here to perform init operations like caching the model in memory    
    """    
    global session    
    global tokenizer    
    global batch_size    
    batch_size = 1  

    INTER_OP_NUM_THREADS = os.getenv("INTER_OP_NUM_THREADS", "None")
    INTRA_OP_NUM_THREADS = os.getenv("INTRA_OP_NUM_THREADS", "None")
    EXECUTION_MODE = os.getenv("EXECUTION_MODE", "None")
    GRAPH_OPTIMIZATION_LEVEL = os.getenv("GRAPH_OPTIMIZATION_LEVEL", "None")
    EXECUTION_PROVIDER = os.getenv("EXECUTION_PROVIDER", "None")
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = int(INTER_OP_NUM_THREADS) if INTER_OP_NUM_THREADS != "None" else 0    
    sess_options.intra_op_num_threads = int(INTRA_OP_NUM_THREADS) if INTRA_OP_NUM_THREADS != "None" else 0    
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel(int(GRAPH_OPTIMIZATION_LEVEL) if GRAPH_OPTIMIZATION_LEVEL != "None" else 99)

    if EXECUTION_MODE == "ExecutionMode.ORT_SEQUENTIAL":
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL    
    elif EXECUTION_MODE == "ExecutionMode.ORT_PARALLEL":
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL    
        
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), os.getenv("MODEL_FOLDER_NAME"), "distilbert-base-cased-distilled-squad.onnx") 
    session = ort.InferenceSession(model_path, sess_options, providers=[EXECUTION_PROVIDER] if EXECUTION_PROVIDER != "None" else ['CPUExecutionProvider'])
    
    logging.info(f"inter_op_num_threads: {sess_options.inter_op_num_threads}")
    logging.info(f"intra_op_num_threads: {sess_options.intra_op_num_threads}")
    logging.info(f"graph_optimization_level: {sess_options.graph_optimization_level}")
    logging.info(f"execution_mode: {sess_options.execution_mode}")

    trained_checkpoint = "distilbert-base-cased-distilled-squad"    
    tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)

    logging.info("Init complete")


def preprocess_example(example, tokenizer):
    max_length = 128    
    stride = 128    
    inputs = tokenizer(
        example["context"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return inputs


def run(raw_data):
    logging.info("distilbert: request received")

    test_data = json.loads(raw_data)["data"]
    # print(json.dumps(test_data, indent=2))    
    # Use read_squad_examples method from run_onnx_squad to read the input file    
    input_data = preprocess_example(test_data, tokenizer)
    for input_meta in session.get_inputs():
        print(input_meta)
    data = {"input_ids": [input_data["input_ids"]],
            "attention_mask": [input_data["attention_mask"]]}
    
    start = time.time()
    result = session.run([], data)
    duration = (time.time() - start) * 1000    
    logging.info("Request processed")
    return duration
