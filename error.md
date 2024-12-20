fatal: unable to access 'https://github.com/song-xudong/gittest.git/': Failed to connect to 127.0.0.1 port 7890 after 2060 ms: Could not connect to server

错误原因为网络代理问题
校园网无法git push，使用个人热点进行提交
还是无法提交参考下面网站
## https://blog.csdn.net/qq_43546721/article/details/139506583
#使用下面命令查看网络问题
ping github.com
#如果没问题使用下面命令即可
git config --global --unset http.proxy 
git config --global --unset https.proxy

