服务器下载文件有点慢，只下载了部分文件
**modify_opt-350m_my.py**为修改opt-350m的核心文件
在开头加入`from modify_opt-350m_my import convert_kvcache_opt_test`，将`(model, config)`作为参数传入即可修改opt-350m的模型
