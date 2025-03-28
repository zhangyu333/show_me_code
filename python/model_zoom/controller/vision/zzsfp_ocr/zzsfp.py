import re
import cv2
import numpy as np
from controller.vision.zzsfp_ocr.predict_system import TextSystem
from controller.vision.zzsfp_ocr.utility import draw_ocr_box_result
from common.oss import OSS
from common.utils import Util
from common.file_utils import clearCache

image_oss = OSS("hz-images")

text_sys = TextSystem()


def padIm(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    blank_im = np.ones((640, 640, 3), dtype=np.uint8) * 114
    r = max(h, w)
    nh, nw = (640, int(w * 640 / r)) if r == h else (int(h * 640 / r), 640)
    new_im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blank_im[int((640 - nh) / 2):int((640 - nh) / 2) + nh,
    int((640 - nw) / 2):int((640 - nw) / 2) + nw] = new_im
    return blank_im


def predict(filename):
    img = cv2.imread(filename)
    padimg = padIm(img)
    pad_local_path = Util.generate_temp_file_path("png")
    cv2.imwrite(pad_local_path, padimg)
    pad_remote_url = image_oss.upload(pad_local_path)
    clearCache(pad_local_path)

    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    pad = 200
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    res = text_sys.detect_and_ocr(img, box_thresh=0.5, unclip_ratio=2)
    my_result = {"info": ""}
    draw_results = []

    bank_accounts = []
    company_names = []
    tins = []
    address_phones = []

    total_box_result = None
    for boxed_result in res:
        ocr_text = boxed_result.ocr_text
        if "代码" in ocr_text:
            invoice_code = ocr_text.split("代码")[-1]
            invoice_code = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', invoice_code)
            my_result["发票代码"] = invoice_code
            draw_results.append(boxed_result)
        if "号码" in ocr_text:
            invoice_number = ocr_text.split("号码")[-1]
            invoice_number = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', invoice_number)
            my_result["发票号码"] = invoice_number
            draw_results.append(boxed_result)
        if "日期" in ocr_text:
            invoice_data = ocr_text.split("日期")[-1]
            invoice_data = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', invoice_data)
            my_result["开票日期"] = invoice_data
            draw_results.append(boxed_result)
        if "开户行及账号" in ocr_text:
            bank_account = ocr_text.split("开户行及账号")[-1]
            bank_account = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', bank_account)
            bank_account_position = boxed_result.box
            center_bank_account_position_y = sum([point[1] for point in bank_account_position]) / 4
            bank_accounts.append({
                "bank_account": bank_account,
                "center_bank_account_position_y": center_bank_account_position_y
            })
            draw_results.append(boxed_result)

        if "称：" in ocr_text:
            company_name = ocr_text.split("称：")[-1]
            company_name = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', company_name)
            company_name_position = boxed_result.box
            center_company_name_position_y = sum([point[1] for point in company_name_position]) / 4
            company_names.append({
                "company_name": company_name,
                "center_company_name_position_y": center_company_name_position_y
            })
            draw_results.append(boxed_result)

        if "纳税人识别号" in ocr_text:
            tin = ocr_text.split("纳税人识别号")[-1]
            tin = re.sub('([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])', '', tin)
            tin_position = boxed_result.box
            center_tin_position_y = sum([point[1] for point in tin_position]) / 4
            tins.append({
                "tin": tin,
                "center_tin_position_y": center_tin_position_y
            })
            draw_results.append(boxed_result)

        if "地址、电话" in ocr_text:
            address_phone = ocr_text.split("地址、电话")[-1]
            address_phone = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', address_phone)
            address_phone_position = boxed_result.box
            center_address_phone_position_y = sum([point[1] for point in address_phone_position]) / 4
            address_phones.append({
                "address_phone": address_phone,
                "center_address_phone_position_y": center_address_phone_position_y
            })
            draw_results.append(boxed_result)

        if "名" == ocr_text:
            draw_results.append(boxed_result)

        if "¥" in ocr_text:
            total = float(ocr_text.split("¥")[-1])
            if "合计" not in my_result.keys():
                my_result["合计"] = total
                total_box_result = boxed_result
            else:
                my_result_total = my_result["合计"]
                if total > my_result_total:
                    my_result["合计"] = total
                    total_box_result = boxed_result
    draw_results.append(total_box_result)
    if len(bank_accounts) == 2:
        bank_account1 = bank_accounts[0]
        bank_account2 = bank_accounts[1]
        if bank_account1.get("center_bank_account_position_y") > bank_account2.get("center_bank_account_position_y"):
            my_result["销售方开户行及账号"] = bank_account2.get("bank_account")
            my_result["购买方开户行及账号"] = bank_account1.get("bank_account")
        else:
            my_result["销售方开户行及账号"] = bank_account2.get("bank_account")
            my_result["购买方开户行及账号"] = bank_account1.get("bank_account")
    else:
        my_result["my_result"] += "开户行及账号信息结构化失败 "

    if len(company_names) == 2:
        company_name1 = company_names[0]
        company_name2 = company_names[1]
        if company_name1.get("center_company_name_position_y") > company_name2.get("center_company_name_position_y"):
            my_result["销售方名称"] = company_name2.get("company_name")
            my_result["购买方名称"] = company_name1.get("company_name")
        else:
            my_result["销售方名称"] = company_name2.get("company_name")
            my_result["购买方名称"] = company_name1.get("company_name")
    else:
        my_result["my_result"] += "名称信息结构化失败 "
    if len(tins) == 2:
        tin1 = tins[0]
        tin2 = tins[1]
        if tin1.get("center_tin_position_y") > tin2.get("center_tin_position_y"):
            my_result["销售方纳税人识别号"] = tin1.get("tin")
            my_result["购买方纳税人识别号"] = tin2.get("tin")
        else:
            my_result["购买方纳税人识别号"] = tin2.get("tin")
            my_result["销售方纳税人识别号"] = tin1.get("tin")
    else:
        my_result["my_result"] += "纳税人识别号信息结构化失败 "

    if len(address_phones) == 2:
        address_phone1 = address_phones[0]
        address_phone2 = address_phones[1]
        if address_phone1.get("center_address_phone_position_y") > address_phone2.get(
                "center_address_phone_position_y"):
            my_result["销售方地址电话"] = address_phone1.get("address_phone")
            my_result["购买方地址电话"] = address_phone2.get("address_phone")
        else:
            my_result["销售方地址电话"] = address_phone2.get("address_phone")
            my_result["购买方地址电话"] = address_phone1.get("address_phone")
    else:
        my_result["my_result"] += "地址电话信息结构化失败 "

    draw_img = draw_ocr_box_result(img, draw_results, 0.5)
    draw_img_h, draw_img_w = draw_img.shape[:2]
    r = max(draw_img_h, draw_img_w)
    nh, nw = (640, int(draw_img_w * 640 / r)) if r == draw_img_h else (int(draw_img_h * 640 / r), 640)
    new_draw_img = cv2.resize(draw_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    local_path = Util.generate_temp_file_path("png")
    cv2.imwrite(local_path, new_draw_img)
    remote_url = image_oss.upload(local_path)
    clearCache(local_path)
    RES = {
        "datalist": my_result,
        "detect_remote_url": remote_url,
        "pad_remote_url": pad_remote_url
    }
    return RES


if __name__ == "__main__":
    filename = "img.png"
    my_result = predict(filename)
    print(my_result)
