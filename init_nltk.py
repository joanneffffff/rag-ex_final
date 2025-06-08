import os
import nltk
import ssl

def download_nltk_data():
    """下载NLTK必要的数据包"""
    print("开始下载NLTK数据...")
    
    # 创建数据目录
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # 设置SSL上下文（如果需要）
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # 需要下载的数据包
    required_packages = [
        'wordnet',
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    # 下载数据包
    for package in required_packages:
        try:
            print(f"正在下载 {package}...")
            nltk.download(package, quiet=True)
            print(f"成功下载 {package}")
        except Exception as e:
            print(f"下载 {package} 时出错: {str(e)}")
    
    print("\nNLTK数据下载完成！")

if __name__ == "__main__":
    download_nltk_data() 