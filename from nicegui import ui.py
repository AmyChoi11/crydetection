

from nicegui import ui

iframe = '<iframe width="360" height="300" src="http://172.20.10.2/" allowfullscreen></iframe> '

@ui.page('/')
def main():
    ui.add_body_html(iframe)
    ui.label('Your baby is tired!').style('color: #Ff8c00; font-size: 200%; font-weight: 300')

    
    
if __name__ in {"__main__", "__mp_main__"}:
    ui.run();