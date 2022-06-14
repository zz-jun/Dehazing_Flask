import base64
import io
from io import BytesIO

import cv2
import requests as req
from PIL import Image
from flask import Flask, request, render_template

import dehazing_AODNet as service_AODNet
import dehazing_FFA as service_FFA
import dehazing_GCANet as service_GCANet

# flask web service
from PSNR import getpsnr
from SSIM import calculate_ssim

app = Flask(__name__, template_folder="web")


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/index')
def index():
    print("********* index ********")
    return render_template('index.html')


@app.route('/detect/imageDetect_psnr', methods=['post'])
def psnr():
    str = request.form.get('psnr')
    select=request.form.get('select')
    print("select:",select)
    print(str)
    if (str == "AODNet_psnr"):

        if(select=="1"):
            gt = cv2.imread('test/clear/0023_2.png')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_1.png')
        elif(select=="2"):
            gt = cv2.imread('test/clear/0098_2.png')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_2.png')
        elif (select == "3"):
            gt = cv2.imread('test/clear/1909_2.png')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_3.png')
        else:
            gt = cv2.imread('test/clear/1_2.jpg')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_4.png')

        ppsnr = getpsnr(gt, img)
    elif (str == "GCANet_psnr"):
        if (select == "1"):
            gt = cv2.imread('test/clear/0023_2.png')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_1.png')
        elif (select == "2"):
            gt = cv2.imread('test/clear/0098_2.png')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_2.png')
        elif (select == "3"):
            gt = cv2.imread('test/clear/1909_2.png')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_3.png')
        else:
            gt = cv2.imread('test/clear/1_2.jpg')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_4.png')

        ppsnr = getpsnr(gt, img)
    else:
        if (select == "1"):
            gt = cv2.imread('test/clear/0023_2.png')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_1.png')
        elif (select == "2"):
            gt = cv2.imread('test/clear/0098_2.png')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_2.png')
        elif (select == "3"):
            gt = cv2.imread('test/clear/1909_2.png')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_3.png')
        else:
            gt = cv2.imread('test/clear/1_2.jpg')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_4.png')

        ppsnr = getpsnr(gt, img)

    print(ppsnr)
    p = "%.3f" % ppsnr
    print(p)
    return p


@app.route('/detect/imageDetect_ssim', methods=['post'])
def ssim():
    str = request.form.get('ssim')
    print(str)
    select = request.form.get('select')
    print("select:", select)
    if (str == "AODNet_ssim"):
        if (select == "1"):
            gt = cv2.imread('test/clear/0023_2.png')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_1.png')
        elif (select == "2"):
            gt = cv2.imread('test/clear/0098_2.png')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_2.png')
        elif (select == "3"):
            gt = cv2.imread('test/clear/1909_2.png')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_3.png')
        else:
            gt = cv2.imread('test/clear/1_2.jpg')
            img = cv2.imread('test/psnr_ssim_img/AOD_Net/img_4.png')

        ssim = calculate_ssim(gt, img)
    elif (str == "GCANet_ssim"):
        if (select == "1"):
            gt = cv2.imread('test/clear/0023_2.png')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_1.png')
        elif (select == "2"):
            gt = cv2.imread('test/clear/0098_2.png')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_2.png')
        elif (select == "3"):
            gt = cv2.imread('test/clear/1909_2.png')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_3.png')
        else:
            gt = cv2.imread('test/clear/1_2.jpg')
            img = cv2.imread('test/psnr_ssim_img/GCANet/img_4.png')
        ssim = calculate_ssim(gt, img)
    else:
        if (select == "1"):
            gt = cv2.imread('test/clear/0023_2.png')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_1.png')
        elif (select == "2"):
            gt = cv2.imread('test/clear/0098_2.png')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_2.png')
        elif (select == "3"):
            gt = cv2.imread('test/clear/1909_2.png')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_3.png')
        else:
            gt = cv2.imread('test/clear/1_2.jpg')
            img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_4.png')

        ssim = calculate_ssim(gt, img)

    print(ssim)
    s = "%.4f" % ssim
    print(s)
    return s


@app.route('/detect/imageDetect_h', methods=['post'])
def upload_h():
    print("获取原始图片")
    file = request.form.get('imageBase64Code_h')
    image_link = request.form.get("imageLink_h")
    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(base64.b64decode(file)))

    img_byte_array = io.BytesIO()


    image.save(img_byte_array, format="PNG")


    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return image_info


@app.route('/detect/imageDetect_A', methods=['post'])
def upload_A():
    # 原始图片
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image_link = request.form.get("imageLink")

    # print("file:",file)
    # print("image_link:", image_link)

    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(base64.b64decode(file)))

    # print("image:",image)

    # step 2. detect image
    image_AODNet = service_AODNet.detect(image)

    # step 3. convert image_array to byte_array

    # img = Image.fromarray(image_array, 'RGB')
    img_byte_array_A = io.BytesIO()
    image_AODNet.save(img_byte_array_A, format='JPEG')

    # step 4. return image_info to page
    image_info_A = base64.b64encode(img_byte_array_A.getvalue()).decode('ascii')
    return image_info_A


@app.route('/detect/imageDetect_G', methods=['post'])
def upload_G():
    # 原始图片
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image_link = request.form.get("imageLink")

    # print("file:",file)
    # print("image_link:", image_link)

    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(base64.b64decode(file)))

    # print("image:",image)

    # step 2. detect image
    image_GCANet = service_GCANet.detect(image)

    # step 3. convert image_array to byte_array

    # img = Image.fromarray(image_array, 'RGB')
    img_byte_array = io.BytesIO()
    image_GCANet.save(img_byte_array, format='JPEG')

    # step 4. return image_info to page
    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return image_info


@app.route('/detect/imageDetect_F', methods=['post'])
def upload_F():
    # 原始图片
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image_link = request.form.get("imageLink")

    # print("file:",file)
    # print("image_link:", image_link)

    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(base64.b64decode(file)))

    # print("image:",image)

    # step 2. detect image
    image_FFA = service_FFA.detect(image)

    # step 3. convert image_array to byte_array

    # img = Image.fromarray(image_array, 'RGB')
    img_byte_array = io.BytesIO()
    image_FFA.save(img_byte_array, format='JPEG')

    # step 4. return image_info to page
    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return image_info


@app.route('/detect/imageDetect', methods=['post'])
def upload():
    # 原始图片
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image_link = request.form.get("imageLink")

    # print("file:",file)
    # print("image_link:", image_link)

    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(BytesIO(base64.b64decode(file)))

    # print("image:",image)

    # step 2. detect image
    image_AODNet = service_AODNet.detect(image)

    # step 3. convert image_array to byte_array

    # img = Image.fromarray(image_array, 'RGB')
    img_byte_array_A = io.BytesIO()
    image_AODNet.save(img_byte_array_A, format='JPEG')

    # step 4. return image_info to page
    image_info_A = base64.b64encode(img_byte_array_A.getvalue()).decode('ascii')
    return image_info_A


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=6006)
