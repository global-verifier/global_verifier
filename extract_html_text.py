#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从HTML文件中提取实际显示在网页上的文本内容
排除script、style、注释等不可见内容
"""

from bs4 import BeautifulSoup, Comment
import sys
import re
from pathlib import Path


def extract_visible_text(html_content):
    """
    从HTML内容中提取可见文本
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除包含Instruction的元素（通过id="instruction-text"定位）
    instruction_div = soup.find(id='instruction-text')
    if instruction_div:
        instruction_div.decompose()
    
    # 移除script和style标签及其内容
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    
    # 移除HTML注释
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    
    # 提取文本
    text = soup.get_text(separator=' ', strip=True)
    
    # 清理多余的空白字符，去除所有换行符（包括字面量\n和真实换行符）
    text = text.replace('\\n', ' ')  # 去除字面量 \n
    text = text.replace('\n', ' ')  # 去除真实换行符
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符合并为单个空格
    text = text.strip()
    
    return text


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_html_text.py <html文件路径> [输出文件路径]")
        print("示例: python extract_html_text.py page.html")
        print("示例: python extract_html_text.py page.html output.txt")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    # 读取HTML文件
    with open(input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 提取文本
    visible_text = extract_visible_text(html_content)
    
    # 输出结果
    if len(sys.argv) >= 3:
        # 保存到文件
        output_file = Path(sys.argv[2])
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(visible_text)
        print(f"文本已保存到: {output_file}")
    else:
        # 输出到控制台
        print(visible_text)


if __name__ == '__main__':
    tmp = """<!DOCTYPE html>\n<html>\n  <head>\n    <link rel="stylesheet" href="/static/style.css">\n    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">\n    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.0.3/css/font-awesome.css\t">\n    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>\n    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.bundle.min.js"></script>\n    <link rel="icon" href="data:,">\n  </head>\n  <body>\n    <div class="container py-5">\n      <div class="row top-buffer">\n        <div class="col-sm-6">\n          <div id="instruction-text" class="text-center">\n            <h4>Instruction:<br>Find me slim fit, machine wash women&#39;s jumpsuits, rompers &amp; overalls with short sleeve, high waist, polyester spandex for daily wear with color: green stripe, and size: large, and price lower than 40.00 dollars</h4>\n          </div>\n        </div>\n      </div>\n      <div class="row top-buffer">\n        <form method="post" action="/?session_id=ayfoedaown">\n          <button type="submit" class="btn btn-success">Back to Search</button>\n        </form>\n      </div>\n      <div class="row top-buffer">\n        <form method="post" action="/?session_id=ayfoedaown&amp;keywords=jacket&amp;page=2">\n          <button type="submit" class="btn btn-primary">&lt; Prev</button>\n        </form>\n      </div>\n      <div class="row top-buffer">\n        <div class="col-md-4 mb-4 mb-md-0">\n          <div class="row top-buffer">\n            <img id="product-image" src="https://m.media-amazon.com/images/I/41D54yBGD-L.jpg" class="item-page-img">\n          </div>\n          \n            <div class="row top-buffer">\n              <h4>size</h4>\n              <div class="radio-toolbar">\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_size0" name="size" value="x-small" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%7D">\n                  <label for="radio_size0">x-small</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_size1" name="size" value="small" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27small%27%7D">\n                  <label for="radio_size1">small</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_size2" name="size" value="medium" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27medium%27%7D">\n                  <label for="radio_size2">medium</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_size3" name="size" value="large" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27large%27%7D">\n                  <label for="radio_size3">large</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_size4" name="size" value="x-large" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-large%27%7D">\n                  <label for="radio_size4">x-large</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_size5" name="size" value="xx-large" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27xx-large%27%7D">\n                  <label for="radio_size5">xx-large</label>\n                \n              </div>\n            </div>\n          \n            <div class="row top-buffer">\n              <h4>color</h4>\n              <div class="radio-toolbar">\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color0" name="color" value="black" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27black%27%7D">\n                  <label for="radio_color0">black</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color1" name="color" value="burgundy" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27burgundy%27%7D">\n                  <label for="radio_color1">burgundy</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color2" name="color" value="charcoal heather" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27charcoal+heather%27%7D">\n                  <label for="radio_color2">charcoal heather</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color3" name="color" value="light heather grey" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27light+heather+grey%27%7D">\n                  <label for="radio_color3">light heather grey</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color4" name="color" value="lilac" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27lilac%27%7D">\n                  <label for="radio_color4">lilac</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color5" name="color" value="navy" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27navy%27%7D">\n                  <label for="radio_color5">navy</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color6" name="color" value="oatmeal heather" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27oatmeal+heather%27%7D">\n                  <label for="radio_color6">oatmeal heather</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color7" name="color" value="olive" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27olive%27%7D">\n                  <label for="radio_color7">olive</label>\n                \n                  \n                  \n                  \n                  <input type="radio" id="radio_color8" name="color" value="white" data-url="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;options=%7B%27size%27%3A+%27x-small%27%2C+%27color%27%3A+%27white%27%7D">\n                  <label for="radio_color8">white</label>\n                \n              </div>\n            </div>\n          \n        </div>\n        <div class="col-md-6">\n          <h2>Amazon Brand - Daily Ritual Women&#39;s 100% Cotton Oversized Fit V-Neck Pullover Sweater</h2>\n          <h4>Price: $20.66 to $29.2</h4>\n          <h4>Rating: N.A.</h4>\n          <div class="row top-buffer">\n            <div class="col-sm-3" name="description">\n              <form method="post" action="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;sub_page=Description&amp;options=%7B%27size%27%3A+%27x-small%27%7D">\n                <button class="btn btn-primary" type="submit">Description</button>\n              </form>\n            </div>\n            <div class="col-sm-3" name="bulletpoints">\n              <form method="post" action="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;sub_page=Features&amp;options=%7B%27size%27%3A+%27x-small%27%7D">\n                <button class="btn btn-primary" type="submit">Features</button>\n              </form>\n            </div>\n            <div class="col-sm-3" name="reviews">\n              <form method="post" action="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;keywords=jacket&amp;page=2&amp;sub_page=Reviews&amp;options=%7B%27size%27%3A+%27x-small%27%7D">\n                <button class="btn btn-primary" type="submit">Reviews</button>\n              </form>\n            </div>\n            \n          </div>\n        </div>\n        <div class="col-sm-2">\n          <div class="row top-buffer">\n            <form method="post" action="/?session_id=ayfoedaown&amp;asin=B07DKGJR74&amp;options=%7B%27size%27%3A+%27x-small%27%7D">\n              <button type="submit" class="btn btn-lg purchase">Buy Now</button>\n            </form>\n          </div>\n        </div>\n      </div>\n    </div>\n  </body>\n  <script>\n    $(document).ready(function() {\n      $(\'input:radio\').each(function() {\n        //console.log($(this).val());\n        let options = JSON.parse(`{"size": "x-small"}`);\n        let optionValues = $.map(options, function(value, key) { return value });\n        //console.log(optionValues);\n        if (optionValues.includes($(this).val())) {\n          $(this).prop(\'checked\', true);\n\n          let option_to_image = JSON.parse(`{"black": "https://m.media-amazon.com/images/I/31QbEtQ4CKL.jpg", "burgundy": "https://m.media-amazon.com/images/I/41D54yBGD-L.jpg", "charcoal heather": "https://m.media-amazon.com/images/I/41cn5LGMtyL.jpg", "large": null, "light heather grey": "https://m.media-amazon.com/images/I/41egkcVXTgL.jpg", "lilac": "https://m.media-amazon.com/images/I/41BumJwQFYL.jpg", "medium": null, "navy": "https://m.media-amazon.com/images/I/31Q7esJsHKL.jpg", "oatmeal heather": "https://m.media-amazon.com/images/I/31K04r2VIyL.jpg", "olive": "https://m.media-amazon.com/images/I/41EefABNmAL.jpg", "small": null, "white": "https://m.media-amazon.com/images/I/31022GR-kSL.jpg", "x-large": null, "x-small": null, "xx-large": null}`);\n//          console.log($(this).val());\n//          console.log(options);\n//          console.log(option_to_image);\n          let image_url = option_to_image[$(this).val()];\n\n          //console.log(image_url);\n          if (image_url) {\n            $("#product-image").attr("src", image_url);\n          }\n        }\n        \n        // reload with updated options\n        this.addEventListener("click", function() {\n          window.location.href = this.dataset.url;\n        });\n\n      });\n    });\n  </script>\n</html>"""
    print(extract_visible_text(tmp))

