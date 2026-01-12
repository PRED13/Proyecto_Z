from django.shortcuts import render
from .utils import *

def index(request):
    return render(request, 'dashboard/index.html', {
        'fotos_06': generate_visualizations_archivo_06(),
        'fotos_07': generate_visualizations_archivo_07(),
        'datos_08': generate_data_processing_08(),
        'datos_05': generate_email_processing_05(),
        'datos_09': generate_pipeline_processing_09(),
        'titulo': 'Dashboard de Análisis NSL-KDD (6, 7, 8, 5, 9)'
    })