import sys
from src.logger import logger


class CustomException(Exception):
    def __init__(self, error_message:str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.error_message_detail(error_message, error_detail)
        logger.error(self.error_message)

    def error_message_detail(self, error_message:str, error_detail:sys) -> str:
        exc_type, exc_value, exc_tb = error_detail.exc_info()
        if exc_tb:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in Python Script:[{file_name}] at line [{line_number}] error:{error_message}"
        else:
            return f"Error: {error_message}"
        
    def __str__(self):
        return self.error_message
