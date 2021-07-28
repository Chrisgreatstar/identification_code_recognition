import random
from PIL import Image
import requests

REQUEST_PREFIX = "https://agrygl.gdcic.net/v1/api/open/getvalidcode?key=examsigning&v="
SAVE_DIR = "codes/"


for i in range(100):
    random_v = str(random.random())
    url = REQUEST_PREFIX + random_v
    # print(url)
    im = Image.open(requests.get(url, stream=True).raw)
    im.save(SAVE_DIR + str(i) + '.jpg')
