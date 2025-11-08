"""
State 处理器 - 将原始环境状态转换为适合存储的经验格式
"""

from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
import re


class StateProcessor:
    """处理环境状态，提取关键信息用于经验存储"""
    
    @staticmethod
    def process_state(raw_state: Dict) -> Dict[str, Any]:
        """
        将原始 state 转换为标准化的经验格式
        
        Args:
            raw_state: 原始状态字典，包含 html, instruction_text, url
            
        Returns:
            处理后的状态字典，包含：
            - page_type: 页面类型（home/search_results/product_page）
            - key_elements: 关键元素列表
            - content_summary: 内容摘要
            - url_pattern: URL 模式（去除 session_id 和动态参数）
        """
        html = raw_state.get('html', '')
        url = raw_state.get('url', '')
        
        # 解析 HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # 1. 识别页面类型
        page_type = StateProcessor._identify_page_type(soup, url)
        
        # 2. 提取关键交互元素
        key_elements = StateProcessor._extract_key_elements(soup, page_type)
        
        # 3. 提取内容摘要（产品信息、价格等）
        content_summary = StateProcessor._extract_content_summary(soup, page_type)
        
        # 4. 标准化 URL（去除 session_id）
        url_pattern = StateProcessor._normalize_url(url)
        
        return {
            'page_type': page_type,
            'key_elements': key_elements,
            'content_summary': content_summary,
            'url_pattern': url_pattern,
            # 保留部分原始信息用于调试
            'url': url_pattern,
        }
    
    @staticmethod
    def _identify_page_type(soup: BeautifulSoup, url: str) -> str:
        """识别页面类型"""
        # 通过 URL 路径判断
        if '/search_results/' in url:
            return 'search_results'
        elif '/item/' in url or 'asin=' in url:
            return 'product_page'
        elif '/cart' in url or '/checkout' in url:
            return 'cart_or_checkout'
        else:
            # 通过 HTML 结构判断
            search_input = soup.find(id='search_input')
            products = soup.find_all(class_='searched-product')
            
            if search_input and not products:
                return 'home'
            elif products:
                return 'search_results'
            else:
                return 'unknown'
    
    @staticmethod
    def _extract_key_elements(soup: BeautifulSoup, page_type: str) -> Dict[str, Any]:
        """提取关键交互元素"""
        elements = {
            'has_search_bar': False,
            'clickable_items': [],
            'navigation_items': [],
        }
        
        # 搜索框
        search_input = soup.find(id='search_input')
        elements['has_search_bar'] = search_input is not None
        
        # 可点击元素
        buttons = soup.find_all(class_='btn')
        product_links = soup.find_all(class_='product-link')
        clickables = []
        
        for btn in buttons:
            text = btn.get_text(strip=True).lower()
            if text and text not in ['search', '']:
                clickables.append(text)
        
        for link in product_links:
            text = link.get_text(strip=True)
            if text:
                clickables.append(text)
        
        elements['clickable_items'] = list(set(clickables))[:20]  # 限制数量
        
        # 导航元素（分页、返回等）
        nav_elements = []
        if soup.find(string=re.compile(r'Next|Previous|Back', re.I)):
            nav_elements.append('has_navigation')
        
        elements['navigation_items'] = nav_elements
        
        return elements
    
    @staticmethod
    def _extract_content_summary(soup: BeautifulSoup, page_type: str) -> Dict[str, Any]:
        """提取内容摘要"""
        summary = {
            'product_count': 0,
            'price_range': None,
            'categories': [],
            'key_text': [],
        }
        
        if page_type == 'search_results':
            # 提取产品数量
            products = soup.find_all(class_='searched-product')
            summary['product_count'] = len(products)
            
            # 提取价格范围
            prices = []
            for product in products:
                price_elem = product.find(class_='product-price')
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    # 提取数字
                    price_match = re.search(r'\$?(\d+\.?\d*)', price_text)
                    if price_match:
                        try:
                            prices.append(float(price_match.group(1)))
                        except:
                            pass
            
            if prices:
                summary['price_range'] = {
                    'min': min(prices),
                    'max': max(prices),
                    'avg': sum(prices) / len(prices)
                }
            
            # 提取产品标题关键词（前3个）
            titles = []
            for product in products[:3]:
                title_elem = product.find(class_='product-title')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    # 提取关键词（前几个词）
                    words = title.split()[:5]
                    titles.append(' '.join(words))
            summary['key_text'] = titles
        
        elif page_type == 'product_page':
            # 产品详情页：提取关键属性
            title_elem = soup.find(class_='product-title')
            price_elem = soup.find(class_='product-price')
            
            if title_elem:
                summary['key_text'].append(title_elem.get_text(strip=True)[:100])
            if price_elem:
                summary['key_text'].append(price_elem.get_text(strip=True))
        
        return summary
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """标准化 URL，去除 session_id 等动态参数"""
        if not url:
            return ''
        
        # 移除 URL 参数中的 session_id
        url = re.sub(r'[?&]session_id=[^&]*', '', url)
        
        # 处理 URL 路径中的 session_id
        # 模式1: /ayfoedaown -> /XXX
        url = re.sub(r'^([^:]+://[^/]+)/([a-z0-9]{8,20})(?:/|$)', r'\1/XXX/', url)
        
        # 模式2: /search_results/ayfoedaown/keyword/page -> /search_results/XXX/keyword/page
        url = re.sub(r'/search_results/([a-z0-9]{8,20})/', r'/search_results/XXX/', url)
        
        # 将 ASIN 等参数泛化
        url = re.sub(r'asin=[A-Z0-9]+', 'asin=XXX', url)
        url = re.sub(r'keywords=[^&]+', 'keywords=XXX', url)
        url = re.sub(r'page=\d+', 'page=N', url)
        
        # 移除端口号（本地开发环境）
        url = re.sub(r':\d+/', '/', url)
        url = re.sub(r':\d+$', '', url)
        
        # 规范化路径中的数字页码
        url = re.sub(r'/(\d+)(?:/|$)', r'/N\1/', url)
        
        return url.strip('/?&')
    
    @staticmethod
    def state_to_string(processed_state: Dict) -> str:
        """
        将处理后的状态转换为字符串表示（用于相似度匹配）
        
        Args:
            processed_state: 处理后的状态字典
            
        Returns:
            字符串表示
        """
        parts = []
        
        # 页面类型
        parts.append(f"PageType:{processed_state['page_type']}")
        
        # 关键元素
        elements = processed_state['key_elements']
        if elements['has_search_bar']:
            parts.append("HasSearch")
        if elements['clickable_items']:
            # 只取前5个关键元素
            items = ','.join(elements['clickable_items'][:5])
            parts.append(f"Clickables:[{items}]")
        
        # 内容摘要
        summary = processed_state['content_summary']
        if summary['product_count'] > 0:
            parts.append(f"Products:{summary['product_count']}")
        if summary['price_range']:
            pr = summary['price_range']
            parts.append(f"PriceRange:${pr['min']:.2f}-${pr['max']:.2f}")
        if summary['key_text']:
            # 只取前2个关键文本
            key_text = '|'.join(summary['key_text'][:2])
            parts.append(f"KeyText:[{key_text[:100]}]")
        
        return " | ".join(parts)


def process_state_for_experience(raw_state: Dict, include_instruction: bool = False) -> Dict:
    """
    为经验存储处理状态（用户接口）
    
    Args:
        raw_state: 原始状态（包含 html, instruction_text, url）
        include_instruction: 是否在状态中包括 instruction_text（不推荐）
        
    Returns:
        适合存储的状态字典
    """
    processor = StateProcessor()
    processed = processor.process_state(raw_state)
    
    # instruction_text 应该存储在 experience 的 confidence 或其他字段中
    # 而不是作为 state 的一部分
    # 因为同样的页面状态可以用于不同的任务
    
    if include_instruction:
        # 仅在特殊情况下包含（不推荐）
        processed['instruction_text'] = raw_state.get('instruction_text', '')
    
    return processed


# 使用示例
if __name__ == "__main__":
    # 测试示例
    test_states = [{
        "html": "<html>...</html>",
        "instruction_text": "Find me...",
        "url": "http://127.0.0.1:3000/search_results/ayfoedaown/jacket/1"
    },
    {
        "html": "<!DOCTYPE html>\n<html>\n  <head>\n    <link rel=\"stylesheet\" href=\"/static/style.css\">\n    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css\">\n    <link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css\">\n    <link rel=\"icon\" href=\"data:,\">\n  </head>\n  <body>\n    <!-- Code reference: https://bootsnipp.com/snippets/m13mN -->\n    <div class=\"container\" style=\"margin-top: 8%;\">\n      <div class=\"col-md-6 col-md-offset-3\">     \n        <div class=\"row\">\n          <div id=\"logo\" class=\"text-center\">\n            <h2>WebShop</h2>\n          </div>\n          <div id=\"instruction-text\" class=\"text-center\">\n            <h4>Instruction: <br>Find me slim fit, machine wash women&#39;s jumpsuits, rompers &amp; overalls with short sleeve, high waist, polyester spandex for daily wear with color: green stripe, and size: large, and price lower than 60.00 dollars</h4>\n          </div>\n          <form role=\"form\" id=\"form-buscar\" method=\"post\" action=\"/?session_id=ayfoedaown\">\n            <div class=\"form-group\">\n              <div class=\"input-group\">\n                <input id=\"search_input\" class=\"form-control\" type=\"text\" name=\"search_query\" placeholder=\"Search...\" required/>\n                <span class=\"input-group-btn\">\n                  <button class=\"btn btn-success\" type=\"submit\"><i class=\"glyphicon glyphicon-search\" aria-hidden=\"true\"></i>Search</button>\n                </span>\n              </div>\n            </div>\n          </form>\n        </div>            \n      </div>\n    </div>\n  </body>\n</html>",
        "instruction_text": "Instruction: Find me slim fit, machine wash women's jumpsuits, rompers & overalls with short sleeve, high waist, polyester spandex for daily wear with color: green stripe, and size: large, and price lower than 60.00 dollars",
        "url": "http://127.0.0.1:3000/ayfoedaown"
    },
    {
        "html": "<!DOCTYPE html>\n<html>\n  <head>\n    <link rel=\"stylesheet\" href=\"/static/style.css\">\n    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css\">\n    <link rel=\"stylesheet\" href=\"https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css\">\n    <link rel=\"icon\" href=\"data:,\">\n  </head>\n  <body>\n    <!-- Code reference: https://bootsnipp.com/snippets/m13mN -->\n    <div class=\"container\" style=\"margin-top: 8%;\">\n      <div class=\"col-md-6 col-md-offset-3\">     \n        <div class=\"row\">\n          <div id=\"logo\" class=\"text-center\">\n            <h2>WebShop</h2>\n          </div>\n          <div id=\"instruction-text\" class=\"text-center\">\n            <h4>Instruction: <br>Find me slim fit, machine wash women&#39;s jumpsuits, rompers &amp; overalls with short sleeve, high waist, polyester spandex for daily wear with color: green stripe, and size: large, and price lower than 50.00 dollars</h4>\n          </div>\n          <form role=\"form\" id=\"form-buscar\" method=\"post\" action=\"/?session_id=ayfoedaown\">\n            <div class=\"form-group\">\n              <div class=\"input-group\">\n                <input id=\"search_input\" class=\"form-control\" type=\"text\" name=\"search_query\" placeholder=\"Search...\" required/>\n                <span class=\"input-group-btn\">\n                  <button class=\"btn btn-success\" type=\"submit\"><i class=\"glyphicon glyphicon-search\" aria-hidden=\"true\"></i>Search</button>\n                </span>\n              </div>\n            </div>\n          </form>\n        </div>            \n      </div>\n    </div>\n  </body>\n</html>",
        "instruction_text": "Instruction: Find me slim fit, machine wash women's jumpsuits, rompers & overalls with short sleeve, high waist, polyester spandex for daily wear with color: green stripe, and size: large, and price lower than 50.00 dollars",
        "url": "http://127.0.0.1:3000/ayfoedaown"
    },
    {
        "html": "<!DOCTYPE html>\n<html>\n  <head>\n    <link rel=\"stylesheet\" href=\"/static/style.css\">\n    <link rel=</div>\n<!--\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-category\">fashion</h5>\n                    </div>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-query\">men&#39;s henleys</h5>\n                    </div>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-product_category\">Clothing, Shoes &amp; Jewelry › Men › Clothing › Active › Active Shirts &amp; Tees › Button-Down Shirts</h5>\n                    </div>\n-->\n                  </div>\n                </div>\n            </ul>\n          </div>\n        </div>\n        \n        <div class=\"col-lg-12 mx-auto list-group-item\">\n          <div class=\"col-lg-4\">\n            <img src=\"https://m.media-amazon.com/images/I/41UXlJTjcBL.jpg\" class=\"result-img\">\n          </div>\n          <div class=\"col-lg-8\">\n            <ul class=\"list-group shadow\">\n                <div class=\"media align-items-lg-center flex-column flex-lg-row p-3\">\n                  <div class=\"media-body order-2 order-lg-1 searched-product\">\n                    \n                    <h4 class=\"mt-0 font-weight-bold mb-2 product-asin\"><a class=\"product-link\" href=\"/?session_id=ayfoedaown&amp;asin=B08DXL22JN&amp;keywords=jacket&amp;page=1&amp;options=%7B%7D\">B08DXL22JN</a></h5>\n                    <h4 class=\"mt-0 font-weight-bold mb-2 product-title\">Cicy Bell Womens Casual Blazers Open Front Long Sleeve Work Office Jackets Blazer</h5>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-price\">$48.99</h6>\n                    </div>\n<!--\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-category\">fashion</h5>\n                    </div>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-query\">women&#39;s suiting &amp; blazers</h5>\n                    </div>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-product_category\">Clothing, Shoes &amp; Jewelry › Women › Clothing › Suiting &amp; Blazers › Blazers</h5>\n                    </div>\n-->\n                  </div>\n                </div>\n            </ul>\n          </div>\n        </div>\n        \n        <div class=\"col-lg-12 mx-auto list-group-item\">\n          <div class=\"col-lg-4\">\n            <img src=\"https://m.media-amazon.com/images/I/51dNdhBtwcL.jpg\" class=\"result-img\">\n          </div>\n          <div class=\"col-lg-8\">\n            <ul class=\"list-group shadow\">\n                <div class=\"media align-items-lg-center flex-column flex-lg-row p-3\">\n                  <div class=\"media-body order-2 order-lg-1 searched-product\">\n                    \n                    <h4 class=\"mt-0 font-weight-bold mb-2 product-asin\"><a class=\"product-link\" href=\"/?session_id=ayfoedaown&amp;asin=B07B9QWJW9&amp;keywords=jacket&amp;page=1&amp;options=%7B%7D\">B07B9QWJW9</a></h5>\n                    <h4 class=\"mt-0 font-weight-bold mb-2 product-title\">1000ft BLACK MADE IN USA RG-11 COMMSCOPE F1177TSEF DIRECT BURIAL TRISHIELD COAXIAL DROP CABLE 14AWG GEL COATED BRAIDS PE JACKET BURIED FLOODED UNDERGROUND COAX CABLE REEL (LOWER SIGNAL LOSS OVER RG6)</h5>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-price\">$330.0</h6>\n                    </div>\n<!--\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-category\">electronics</h5>\n                    </div>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-query\">signal boosters</h5>\n                    </div>\n                    <div class=\"d-flex align-items-center justify-content-between mt-1\">\n                      <h5 class=\"font-weight-bold my-2 product-product_category\">Electronics › Accessories &amp; Supplies › Audio &amp; Video Accessories › Cables &amp; Interconnects › Video Cables › F-Pin-Coaxial Tip</h5>\n                    </div>\n-->\n                  </div>\n                </div>\n            </ul>\n          </div>\n        </div>\n        \n      </div>\n    </div>\n  </body>\n</html>",
        "instruction_text": "Instruction: Find me slim fit, machine wash women's jumpsuits, rompers & overalls with short sleeve, high waist, polyester spandex for daily wear with color: green stripe, and size: large, and price lower than 50.00 dollars",
        "url": "http://127.0.0.1:3000/search_results/ayfoedaown/jacket/1"
    }]
    
    for i in range(len(test_states)):
        state = test_states[i]
        processed = process_state_for_experience(state)
        print(f"[{i}] 处理后的状态:")
        print(processed)

