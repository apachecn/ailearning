# coding: utf-8
import os
import sys


def format_file(filename, str1, str2):
    """
    文件内容的替换功能
    :return:
    """
    with open(filename, 'r') as f:
        var_object = f.read()
        if "gitalk" not in var_object:
            var_object = var_object.replace(str1, str2)
        # print(var_object)

    f = open(filename, "w")
    f.write(var_object)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        version, u_type = sys.argv[1], sys.argv[2]
    else:
        print("Usage: 参数个数为%s - 错误，应该改为3" % len(sys.argv))
        sys.exit(-1)

    tag = True
    if u_type == "index":
        tag = False
        # if version == "home":
        #     filename = "_book/index.html"
        # else:
        #     filename = "_book/docs/%s/index.html" % version
        # str1 = """
        # </head>
        # <body>
        # """

        # str2 = """
        # <script type="text/javascript">
        #     function hidden_left(){
        #         document.getElementsByClassName("btn pull-left js-toolbar-action")[0].click()
        #     }
        #     // window.onload = hidden_left();
        # </script>
        # </head>
        # <body onload="hidden_left()">
        # """
    elif u_type == "book":
        if version == "home":
            filename = "book.json"
            tag = False
        else:
            filename = "docs/%s/book.json" % version
            str1 = "https://github.com/apachecn/pytorch-doc-zh/blob/master"
            str2 = "https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/%s" % version

    elif u_type == "powered":
        if version == "home":
            filename = "node_modules/gitbook-plugin-tbfed-pagefooter/index.js"
        else:
            filename = "docs/%s/node_modules/gitbook-plugin-tbfed-pagefooter/index.js" % version
        str1 = "powered by Gitbook"
        str2 = "由 ApacheCN 团队提供支持"

    elif u_type == "gitalk":
        if version == "home":
            filename = "node_modules/gitbook-plugin-tbfed-pagefooter/index.js"
        else:
            filename = "docs/%s/node_modules/gitbook-plugin-tbfed-pagefooter/index.js" % version
        str1 = """      var str = ' \\n\\n<footer class="page-footer">' + _copy +
        '<span class="footer-modification">' +
        _label +
        '\\n{{file.mtime | date("' + _format +
        '")}}\\n</span></footer>'"""

        str2 = """
      var str = '\\n\\n'+
      '\\n<hr/>'+
      '\\n<div align="center">'+
      '\\n    <p><a href="http://www.apachecn.org" target="_blank"><font face="KaiTi" size="6" color="red">我们一直在努力</font></a></p>'+
      '\\n    <p><a href="https://github.com/apachecn/pytorch-doc-zh/" target="_blank">apachecn/pytorch-doc-zh</a></p>'+
      '\\n    <p><iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=pytorch-doc-zh&type=watch&count=true&v=2" frameborder="0" scrolling="0" width="100px" height="25px"></iframe>'+
      '\\n    <iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=pytorch-doc-zh&type=star&count=true" frameborder="0" scrolling="0" width="100px" height="25px"></iframe>'+
      '\\n    <iframe align="middle" src="https://ghbtns.com/github-btn.html?user=apachecn&repo=pytorch-doc-zh&type=fork&count=true" frameborder="0" scrolling="0" width="100px" height="25px"></iframe>'+
      '\\n    <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=bcee938030cc9e1552deb3bd9617bbbf62d3ec1647e4b60d9cd6b6e8f78ddc03"><img border="0" src="http://data.apachecn.org/img/logo/ApacheCN-group.png" alt="ML | ApacheCN" title="ML | ApacheCN"></a></p>'+
      '\\n</div>'+
      '\\n <div style="text-align:center;margin:0 0 10.5px;">'+
      '\\n     <script async src="//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>'+
      '\\n     <ins class="adsbygoogle"'+
      '\\n         style="display:inline-block;width:728px;height:90px"'+
      '\\n         data-ad-client="ca-pub-3565452474788507"'+
      '\\n         data-ad-slot="2543897000">'+
      '\\n     </ins>'+
      '\\n     <script>(adsbygoogle = window.adsbygoogle || []).push({});</script>'+
      '\\n'+
      '\\n    <script>'+
      '\\n      var _hmt = _hmt || [];'+
      '\\n      (function() {'+
      '\\n        var hm = document.createElement("script");'+
      '\\n        hm.src = "https://hm.baidu.com/hm.js?38525fdac4b5d4403900b943d4e7dd91";'+
      '\\n        var s = document.getElementsByTagName("script")[0]; '+
      '\\n        s.parentNode.insertBefore(hm, s);'+
      '\\n      })();'+
      '\\n    </script>'+
      '\\n'+
      '\\n    <script async src="https://www.googletagmanager.com/gtag/js?id=UA-102475051-10"></script>'+
      '\\n    <script>'+
      '\\n      window.dataLayer = window.dataLayer || [];'+
      '\\n      function gtag(){dataLayer.push(arguments);}'+
      '\\n      gtag(\\'js\\', new Date());'+
      '\\n'+
      '\\n      gtag(\\'config\\', \\'UA-102475051-10\\');'+
      '\\n    </script>'+
     '\\n</div>'+
      '\\n'+
      '\\n<meta name="google-site-verification" content="pyo9N70ZWyh8JB43bIu633mhxesJ1IcwWCZlM3jUfFo" />'+
      '\\n<iframe src="https://www.bilibili.com/read/cv2710377" style="display:none"></iframe>'+ 
      '\\n<img src="http://t.cn/AiCoDHwb" hidden="hidden" />'

      str += '\\n\\n'+
      '\\n<div>'+
      '\\n    <link rel="stylesheet" href="https://unpkg.com/gitalk/dist/gitalk.css">'+
      '\\n    <script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>'+
      '\\n    <script src="https://cdn.bootcss.com/blueimp-md5/2.10.0/js/md5.min.js"></script>'+
      '\\n    <div id="gitalk-container"></div>'+
      '\\n    <script type="text/javascript">'+
      '\\n        const gitalk = new Gitalk({'+
      '\\n        clientID: \\'2e62dee5b9896e2eede6\\','+
      '\\n        clientSecret: \\'ca6819a54656af0d87960af15315320f8a628a53\\','+
      '\\n        repo: \\'pytorch-doc-zh\\','+
      '\\n        owner: \\'apachecn\\','+
      '\\n        admin: [\\'jiangzhonglian\\', \\'wizardforcel\\'],'+
      '\\n        id: md5(location.pathname),'+
      '\\n        distractionFreeMode: false'+
      '\\n        })'+
      '\\n        gitalk.render(\\'gitalk-container\\')'+
      '\\n    </script>'+
      '\\n</div>'

      str += '\\n\\n<footer class="page-footer">' + _copy + '<span class="footer-modification">' + _label + '\\n{{file.mtime | date("' + _format + '")}}\\n</span></footer>'
        """

    # 状态为 True 就进行替换
    if tag: format_file(filename, str1, str2)
