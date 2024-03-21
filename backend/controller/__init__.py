
from flask import Blueprint, request
import service

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/mask-file', methods=['POST'])
def get_mask_file( ):

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'File is missing', 404
    
    scretch_img = request.files['file']
    mask_file = service.get_mask_file(scretch_img)

    return {
        "content":mask_file,
        "message":'성공'
    }, 200