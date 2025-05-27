from nicegui import ui

iframe = '<iframe width="350" height="260" src="http://172.20.10.2/" allowfullscreen></iframe> '

@ui.page('/')
def main():
    
    with ui.column().classes('w-full h-screen flex justify-center items-center'):
        
        with ui.column().classes('flex flex-col items-center'):
            ui.image('https://ibb.co/nMtZfSGC')
            ui.add_body_html(iframe)
            ui.label('Your baby is tired!').classes('text-[#Ff8c00] text-2xl font-light')
            ui.notify('Your baby is tired!', close_button='OK')

if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
