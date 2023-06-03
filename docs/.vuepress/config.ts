import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from '@vuepress/cli'

export default defineUserConfig({
	theme: defaultTheme({
		// Site header
		head: [
			['link', { rel: 'apple-touch-icon', sizes:"180x180", href:'/apple-touch-icon.png' }],
			['link', { rel: 'icon', type:"image/png", sizes:"32x32", href:"/favicon-32x32.png" }],
			['link', { rel: 'icon', type:"image/png", sizes:"16x16", href:"/favicon-16x16.png" }],
			['link', { rel: 'manifest', href:"/site.webmanifest" }]
		],
		
		// Site logo and navbar
		logo: "/img/rwkv-avartar-256p.png",
		navbar: [
			{ text: 'Main Github', link: 'https://github.com/BlinkDL/RWKV-LM' },
			{ text: 'Hugging Face Integration', link: 'https://huggingface.co/docs/transformers/model_doc/rwkv' },
			{ text: 'Community Discord', link: 'https://discord.gg/bDSBUMeFpc' }
		],

		// Sidebar menu
		sidebar: [
			{ text: 'RWKV Lanugage Model', link: '/' },
			{ 
				text: 'Getting Started', 
				link: '/basic/play.md',
				children: [
					"/basic/play.md",
					"/basic/integrate.md",
					"/basic/FAQ.md"
				]
			},
			{ 
				text: 'Advanced', 
				link: '/advance/',
				children: [
					"/advance/finetune.md"
				]
			},
			{
				text: 'Community',
				link: '/community/code-of-conduct.md',
				children: [
					"/community/code-of-conduct.md",
					"/community/contribute.md",
					"/community/links.md"
				]
			}
		]
	})
})