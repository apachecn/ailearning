(function(){
	var cnzzId = window.$docsify.cnzzId
	var unRepo = window.$docsify.repo || ''
	var [un, repo] = unRepo.split('/')
  var footer = `
      <hr/>
      <div align="center">
        <p><a href="https://www.apachecn.org/" target="_blank"><font face="KaiTi" size="6" color="red">我们一直在努力</font></a><p>
        <p><a href="https://github.com/${unRepo}" target="_blank">${unRepo}</a></p>
        <p><a target="_blank" href="https://qm.qq.com/cgi-bin/qm/qr?k=5u_aAU-YlY3fH-m8meXTJzBEo2boQIUs&jump_from=webapi&authKey=CVZcReMt/vKdTXZBQ8ly+jWncXiSzzWOlrx5hybX5pSrKu6s0fvGX54+vHHlgYNt"><img border="0" src="//pub.idqqimg.com/wpa/images/group.png" alt="【布客】中文翻译组" title="【布客】中文翻译组"></a></p>
        <p><span id="cnzz_stat_icon_${cnzzId}"></span></p>
        <p><a href="https://get.brightdata.com/apachecn" target="_blank"><img src="img/ad/partnerstack.gif" /></a><p>
      </div>
      <hr/>
      <!-- 来必力City版安装代码 -->
      <div id="lv-container" data-id="city" data-uid="MTAyMC81ODA2NC8zNDUyNw==">
        <script type="text/javascript">
        (function(d, s) {
            var j, e = d.getElementsByTagName(s)[0];

            if (typeof LivereTower === 'function') { return; }

            j = d.createElement(s);
            j.src = 'https://cdn-city.livere.com/js/embed.dist.js';
            j.async = true;

            e.parentNode.insertBefore(j, e);
        })(document, 'script');
        </script>
      <noscript> 为正常使用来必力评论功能请激活JavaScript</noscript>
      </div>
      <!-- City版安装代码已完成 -->
	`
  var plugin = function(hook) {
    hook.afterEach(function(html) {
      return html + footer
    })
    hook.doneEach(function() {
      (adsbygoogle = window.adsbygoogle || []).push({})
    })
  }
  var plugins = window.$docsify.plugins || []
  plugins.push(plugin)
  window.$docsify.plugins = plugins
})()
