## Setup the tools
* Anaconda -> https://docs.anaconda.com/anaconda/install/
1. Recuerda agregar las variables de entorno de anaconda al path -> abres el anaconda prompt y poner "where conda"
* VS -> https://code.visualstudio.com/download
2. Recuerda instalar el tool de python para que interprete correctamente el environment.
3. Para cambiar a que ejecute con el cmd en vez del power shell ve a file, preferences, settings-> escribe terminal y busca el apartado terminal>integrated>defaulprofile
* git bash -> https://git-scm.com/downloads

## Install environment :rocket:
----------------------
* conda env create -p .\venv -f .\bin\local\environment.yml :relaxed: 
* Para activar el environment indefinidamente -> ctrl+shift+p ->"Terminal: Select Default Profile"

## Github 
----------------------
* git checkout name_rama_interes_remoto **Cambia de rama en el repo local**
* git merge name_rama_remoto  **Fusiona la rama remota con la rama local actual**
* git push origin name_rama_actual_local:name_rama_remota **Hace push de la rama local a la rama remota**
* git add file_name **Agregar un fichero específico antes de un commit**
* git add . **Agrega todos los cambios ante de un commit**
* git commit -m "Mensaje descriptivo del commit" **Crea un commit con ese mensaje**
* git fetch origin **Actualiza el repositorio local, pero no actualiza los ficheros automaticamente**
* git pull origin name_rama_remota **Actualiza inmediatamente el repo local con la rama especificada**

**Gestión de issues y pull request:**

1. Generar un issue en la página web de github
2. Trabajar los cambios sobre la rama que gestionará el issue
3. Agregar los ficheros modificados con el comando git add
4. Hacer commit pero es muy importante incluir en el comentario el número del issue EX: "Resuleve #10" si el issue es el 10
5. Hacer push sobre tu rama (la rama encargada del issue)
6. Crear la pull request (base: Main) (compare : rama encargada del issue)
7. El revisor se encargará de validar que los cambios propuestos estan ok y entonces aceptará cambio y guardara la pull request.

## Developers
----------------------
The developers responsible for the development and maintaining of this project.

* **Julian Caro** - *Author/Maintainer* - [mateocivil10@gmail.com]

