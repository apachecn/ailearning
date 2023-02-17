(function(){
	var cnzzId = window.$docsify.cnzzId
	var unRepo = window.$docsify.repo || ''
	var [un, repo] = unRepo.split('/')
    var footer = `
      <hr/>
      <div align="center">
        <p><a href="http://www.apachecn.org/" target="_blank"><font face="KaiTi" size="6" color="red">我们一直在努力</font></a><p>
        <p><a href="https://github.com/${unRepo}" target="_blank">${unRepo}</a></p>
        <p><a target="_blank" href="https://qm.qq.com/cgi-bin/qm/qr?k=5u_aAU-YlY3fH-m8meXTJzBEo2boQIUs&jump_from=webapi&authKey=CVZcReMt/vKdTXZBQ8ly+jWncXiSzzWOlrx5hybX5pSrKu6s0fvGX54+vHHlgYNt"><img border="0" src="//pub.idqqimg.com/wpa/images/group.png" alt="【布客】中文翻译组" title="【布客】中文翻译组"></a></p>
        <p><span id="cnzz_stat_icon_${cnzzId}"></span></p>
        <p><a href="https://get.brightdata.com/apachecn" target="_blank"><img src="img/ad/partnerstack.gif" /></a><p>
      </div>
      <hr/>
      <div id="gitalk-container" ></div>
	`
  var plugin = function(hook) {
    hook.afterEach(function(html) {
      return html + footer
    })
    hook.doneEach(function() {
      (adsbygoogle = window.adsbygoogle || []).push({})
      new Gitalk(window.$docsify.gitalk)
          .render(window.$docsify.gitalk.container)
    })
  }
  var plugins = window.$docsify.plugins || []
  plugins.push(plugin)
  window.$docsify.plugins = plugins
})()
