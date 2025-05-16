import os
import logging
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig

class MoEProfiler:
    """Base class for profiling Mixture of Experts (MoE) models."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the MoE Profiler.
        
        Args:
            args: Command line arguments with model configuration
        """
        self.args = args
        self.bs = args.batch_size
        self.activated_experts_record = []
        
        # Set up logging
        os.makedirs('logs', exist_ok=True)
        self.logger = self.setup_logger('logs/moe_profiling.log')
        
        # Ensure output directory exists
        self.out_file = os.path.join(args.output_dir, f"{args.model.replace('/', '_')}.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load dataset
        self.load_dataset()
        
        # Load model configuration
        self.load_model_config()

    def setup_logger(self, log_file: str) -> logging.Logger:
        """Set up a file logger that doesn't propagate to stdout.
        
        Args:
            log_file: Path to the log file
            
        Returns:
            Logger instance
        """
        logger = logging.getLogger("file_only_logger")
        logger.setLevel(logging.DEBUG)
        
        if not logger.handlers:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.propagate = False
        
        return logger
    
    def load_dataset(self) -> None:
        """Load the dataset for profiling."""
        self.dataset = pickle.load(open('math_questions.pkl', 'rb'))
        
        # Convert to chat messages
        self.chat_messages = []
        for message in tqdm(self.dataset, desc='Generating chat templates'):
            msg = [{'role': 'user', 'content': message}]
            self.chat_messages.append(msg)
    
    def load_model_config(self) -> None:
        """Load model configuration."""
        model_config = AutoConfig.from_pretrained(
            self.args.model, 
            trust_remote_code=self.args.trust_remote_code
        )
        
        # Override config if specified
        if self.args.hidden_size:
            model_config.hidden_size = self.args.hidden_size
        if self.args.num_layers:
            model_config.num_hidden_layers = self.args.num_layers
            
        self.model_config = model_config.to_dict()
        
        # Fix rope config if needed
        if 'rope_scaling' in self.model_config and self.model_config['rope_scaling']:
            self.model_config['rope_scaling']['rope_type'] = self.model_config['rope_scaling']['type']
    
    def patch_moe_functions(self) -> None:
        """
        Patch the relevant MoE functions to track activated experts.
        
        This method should be implemented by subclasses for each specific framework.
        """
        raise NotImplementedError("Subclasses must implement this method for their specific framework")
    
    def restore_moe_functions(self) -> None:
        """
        Restore the original MoE functions after profiling.
        
        This method should be implemented by subclasses for each specific framework.
        """
        raise NotImplementedError("Subclasses must implement this method for their specific framework")
    
    def initialize_model(self) -> None:
        """
        Initialize the model with the specified configuration.
        
        This method should be implemented by subclasses for each specific framework.
        """
        raise NotImplementedError("Subclasses must implement this method for their specific framework")
    
    def run_inference(self) -> None:
        """
        Run inference on the dataset and collect MoE activation statistics.
        
        This method should be implemented by subclasses for each specific framework.
        """
        raise NotImplementedError("Subclasses must implement this method for their specific framework")
    
    def should_skip_profiling(self) -> bool:
        """Check if profiling for this configuration has already been done."""
        if not os.path.exists(self.out_file):
            return False
            
        df = pd.read_csv(self.out_file)
        return df[(df['dataset'] == 'MATH_4000') & (df['batch_size'] == self.bs)]['average activated experts'].values.size > 0
    
    def save_results(self) -> None:
        """Save the profiling results to a CSV file."""
        if not self.activated_experts_record:
            raise Exception("No activated experts recorded. Batch size might be too large for available GPU memory.")
        
        avg_experts = sum(self.activated_experts_record) / len(self.activated_experts_record)
        print(f"The average number of activated experts of {self.args.model} on batch size = {self.bs} is {avg_experts}")
        
        # Create or update results dataframe
        if os.path.exists(self.out_file):
            df = pd.read_csv(self.out_file)
        else:
            df = pd.DataFrame(columns=['dataset', 'batch_size', 'average activated experts'])
            
        new_row = {
            'dataset': 'MATH_4000', 
            'batch_size': self.bs, 
            'average activated experts': avg_experts
        }
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(self.out_file, index=False)
    
    def run(self) -> None:
        """Run the complete profiling process."""
        if self.should_skip_profiling():
            print(f"Batch size {self.bs} of Dataset MATH already exists in the profiling table")
            return
            
        try:
            # Initialize the model
            self.initialize_model()
            
            # Patch MoE functions to track activation
            self.patch_moe_functions()
            
            # Run inference
            self.run_inference()
            
            # Save results
            self.save_results()
            
        finally:
            # Restore original functions
            self.restore_moe_functions()