from transformers import Trainer, DataCollatorForLanguageModeling
from src.logger import get_logger

logger = get_logger(__name__)

def build_trainer(model, tokenizer, train_dataset, val_dataset, training_args):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info("Building trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    return trainer

def run_training(trainer):
    """starts training and returns the final train results"""
    logger.info("Starting training")
    results = trainer.train()
    logger.info("Training finished")
    return results