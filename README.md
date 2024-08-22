# Sistema de Recomendación Híbrido

Este proyecto implementa un sistema de recomendación híbrido utilizando `Dash` y `Plotly` para la visualización interactiva de datos. El sistema está diseñado para recomendar películas a usuarios basándose en un enfoque de red, que combina recomendaciones basadas en similitudes y en predicciones personalizadas.

## Estructura del Proyecto

### Archivos y Directorios Principales

- **app_**: Archivo principal de la aplicación donde se inicializa el servidor Flask subyacente a Dash.
- **layouts/**: Contiene los layouts de las distintas páginas (`home`, `dashboard`, `aboutus`, etc.).
- **spatial/**: Módulo que gestiona las recomendaciones espaciales.
- **assets/**: Directorio que almacena imágenes, hojas de estilo, y otros recursos estáticos.

### Librerías Principales

- `dash`: Framework principal utilizado para crear la interfaz web interactiva.
- `dash_bootstrap_components`: Componentes Bootstrap para Dash, utilizados en la creación de la barra de navegación y el sidebar.
- `plotly`: Librería utilizada para la creación de gráficos interactivos.
- `networkx`: Utilizada para crear y manipular grafos, que representan las relaciones entre usuarios y películas.
- `psutil`: Monitorea el uso de la memoria en el sistema.

## Funcionalidades Principales

### 1. Navegación

La aplicación cuenta con una barra de navegación superior y un sidebar que permite al usuario moverse entre diferentes secciones:

- **Home**: Página principal con una introducción al sistema de recomendación.
- **Dashboard**: Página que presenta estadísticas y visualizaciones clave.
- **Recomendación**: Página donde el usuario puede generar recomendaciones personalizadas.
- **About Us**: Información sobre el equipo de desarrollo.

### 2. Sistema de Recomendación

El sistema de recomendación permite a los usuarios:

- Generar recomendaciones personalizadas basadas en sus preferencias y comportamiento previo.
- Visualizar un grafo que representa la red de similitudes entre películas.
- Explorar tablas con predicciones y similitudes entre películas.

### 3. Monitoreo del Sistema

La aplicación incluye un monitor de memoria que se actualiza cada segundo, mostrando el uso de la memoria RAM del sistema en la interfaz.

## Cómo Ejecutar el Proyecto

### Prerrequisitos

- Python 3.7 o superior
- Pip para la instalación de dependencias

### Instalación

1. Clona este repositorio en tu máquina local:
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd tu_repositorio
    ```

2. Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

3. Ejecuta la aplicación:
    ```bash
    python app_.py
    ```

La aplicación estará disponible en `http://0.0.0.0:8000`.

## Personalización

El proyecto está diseñado para ser fácilmente extensible. Puedes agregar nuevas páginas o modificar las existentes editando los archivos en el directorio `layouts/`. Los callbacks de Dash permiten actualizar dinámicamente el contenido en función de la interacción del usuario.

## Contribuciones

Si deseas contribuir a este proyecto, por favor, crea un fork del repositorio y envía tus cambios a través de un pull request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

