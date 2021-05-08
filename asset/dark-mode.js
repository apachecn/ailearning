document.addEventListener('DOMContentLoaded', function() {
	var style = document.querySelector('#invert')
	if (style == null) {
		style = document.createElement('style')
		style.id = 'invert'
		document.head.append(style)
	}
	var btn = document.querySelector('#dark-mode-btn')
	if (btn == null) {
		btn = document.createElement('div')
		btn.id = 'dark-mode-btn'
		btn.classList.add('light-logo')
		document.body.append(btn)
	}
	
	var enableDarkMode = function() {
		style.innerText = 'html,img,pre,#dark-mode-btn{filter:invert(100%)}'
		btn.classList.remove('light-logo')
		btn.classList.add('dark-logo')
		localStorage.darkLight = 'dark'
		
	}
	var disableDarkMode = function() {
		style.innerText = ''		
		btn.classList.remove('dark-logo')
		btn.classList.add('light-logo')
		localStorage.darkLight = 'light'
	}
	
	btn.addEventListener('click', function(){
		var currMode = localStorage.darkLight || 'light'
		if (currMode == 'light')
			enableDarkMode()
		else 
			disableDarkMode()
	})
	
	if (localStorage.darkLight == 'dark')
		enableDarkMode()
	
})

