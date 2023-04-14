import sys
import logging

def error_message_detail(error, error_deatils:sys):
    _,_,exc_tb = error_deatils.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error has occured with name [{0}] + at line number [{1}] + with message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)

    def __str__(self) -> str:
        return self.error_message
    