new_installed_apps = """INSTALLED_APPS = [
    'alliance_admin',"""

with open('studio/settings.py', 'r+') as f:
    contents = f.read().replace("INSTALLED_APPS = [", new_installed_apps)
    f.seek(0)
    f.truncate()
    f.write(contents)

new_urls = """urlpatterns = [
    path('', include('alliance_admin.urls', namespace='alliance_admin')),"""

with open('studio/urls.py', 'r+') as f:
    contents = f.read().replace("urlpatterns = [", new_urls)
    f.seek(0)
    f.truncate()
    f.write(contents)

alliance_admin_button = """<li class="nav-item list-group-item list-group-item-action sidebar-item">
            <img src="{% static '/icons/toggles.svg' %}" alt="" width="24" height="24" title="Bootstrap"
                 class="side-bar-icon">
            <a class="nav-link side-bar-text" href="{% url 'projects:settings' request.user project.slug %}">
                Settings
            </a>
        </li>
        <li class="nav-item list-group-item list-group-item-action sidebar-item">
            <img src="{% static '/icons/person.svg' %}" alt="" width="24" height="24" title="Bootstrap"
                 class="side-bar-icon">
            <a class="nav-link side-bar-text" href="{% url 'alliance_admin:index' request.user project.slug %}">
                Alliance
            </a>
        </li>"""

with open('templates/baseproject.html', 'r+') as f:
    contents = f.read().replace("""<li class="nav-item list-group-item list-group-item-action sidebar-item">
            <img src="{% static '/icons/toggles.svg' %}" alt="" width="24" height="24" title="Bootstrap"
                 class="side-bar-icon">
            <a class="nav-link side-bar-text" href="{% url 'projects:settings' request.user project.slug %}">
                Settings
            </a>
        </li>""", alliance_admin_button)
    f.seek(0)
    f.truncate()
    f.write(contents)

alliance_admin_migrations = "python3 manage.py makemigrations ingress datasets deployments experiments files labs models projects reports workflows alliance_admin"

with open('scripts/run_web.sh', 'r+') as f:
    contents = f.read().replace("python3 manage.py makemigrations ingress datasets deployments experiments files labs models projects reports workflows", alliance_admin_migrations)
    f.seek(0)
    f.truncate()
    f.write(contents)
