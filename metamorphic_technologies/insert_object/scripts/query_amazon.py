import boto3
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor
import sys
from botocore.config import Config

my_config = Config(
    region_name = 'us-east-1',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)


if __name__ == "__main__":
     
    # path="kite.jpg"
    # path="kite.png"
    path=sys.argv[1]
    # path="new_1.png"
    client=boto3.client('rekognition')


    image = Image.open(open(path,'rb'))
    row,col = image.size
    stream = io.BytesIO()
    image.save(stream, format=image.format)
    image_binary = stream.getvalue()

    client = boto3.client('rekognition', config=my_config, aws_access_key_id = 'AKIAJHRH733W6CNWBKEQ',
        aws_secret_access_key = '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', endpoint_url= r"https://rekognition-requester.us-east-1.amazonaws.com/")
    response = client.detect_labels(Image={'Bytes': image_binary})
    # MaxLabels=10)

    imgWidth, imgHeight = image.size  
    draw = ImageDraw.Draw(image)  
                    
    # calculate and display bounding boxes for each detected face       
    for label in response['Labels']:                       
        for box in label['Instances']:                

            print (label['Name'])
            left = imgWidth * box['BoundingBox']['Left']
            top = imgHeight * box['BoundingBox']['Top']
            width = imgWidth * box['BoundingBox']['Width']
            height = imgHeight * box['BoundingBox']['Height']
                    
            points = (
                (left,top),
                (left + width, top),
                (left + width, top + height),
                (left , top + height),
                (left, top)

            )
            # if label['Name'] == "Person":
            draw.line(points, fill='#00d400', width=2)

        # Alternatively can draw rectangle. However you can't set line width.
        #draw.rectangle([left,top, left + width, top + height], outline='#00d400') 




    image.show()


