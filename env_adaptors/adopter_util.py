from bs4 import BeautifulSoup, Comment
import re

def extract_visible_text(html_content):
    """
    从HTML内容中提取可见文本
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除包含Instruction的元素（通过id="instruction-text"定位） 用于 webshop 环境
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