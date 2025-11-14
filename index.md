---
layout: page
title: "Home"
class: home
---

<!-- # About Me -->

<div class="columns" markdown="1">

<div class="intro" markdown="1">
My research interest lies in the area of machine learning theory, quantum information theory and topics relating to quantum machine learning & cryptography. 
<!-- Recently my research focuses on multi-modal learning, diffusion generative models, medical imaging, and geospatial AI. -->

I have a Master's degree in Machine Learning, a second Master's in Mathematical Sciences, both from the [African Institute for Mathematical Sciences](https://aimsammi.org/) (AIMS), and a Bachelor's degree in Mathematics from the [Kwame Nkrumah University of Science and Technology](https://www.knust.edu.gh/) (KNUST). 

**<span style="color: #ffc000;">I am open to collaboration and actively seeking PhD opportunities. If you're interested in working together or have opportunities available, please reach out to <a href="mailto:princemensah@aims.edu.gh" style="color: #3eb7f0;">me!</a></span>**

<p class="social-links">
  <!-- <a href="https://scholar.google.com/citations?user=Kq0dhLAAAAAJ&hl"><i class="ai ai-google-scholar-square"></i> Google Scholar</a> &nbsp;&bull;&nbsp; -->
  <!-- <a href="https://twitter.com/ChenyuW64562111"><i class="fab fa-twitter"></i> Twitter</a> &nbsp;&bull;&nbsp; -->
  <a href="assets/princemensah_resume.pdf"><i class="far fa-file-alt"></i> Resume</a> &nbsp;&bull;&nbsp;
  <!-- <a href="https://github.com/princeemensah"><i class="fab fa-github"></i> GitHub</a> &nbsp;&bull;&nbsp; -->
  <a href="https://www.linkedin.com/in/prince-mensah/"><i class="fab fa-linkedin"></i> LinkedIn</a> &nbsp;&bull;&nbsp;
  <a href="mailto:princemensah@aims.edu.gh"><i class="fas fa-envelope"></i> Email</a>
</p>
</div>

<div class="me" markdown="1">
<picture>
  <img
    src='/images/profile_pic.jpg'
    alt='Prince Mensah'>
</picture>

{:.no-list}
* <a href="mailto:princemensah@aims.edu.gh">princemensah@aims.edu.gh</a>
</div>

</div>

## Featured <a href="{{ "/projects/" | relative_url }}">Projects</a>

<div class="featured-projects">
  {% assign sorted_projects = site.data.projects | sort: 'highlight' %}
  {% for project in sorted_projects %}
    {% if project.highlight %}
      {% include project.html project=project %}
    {% endif %}
  {% endfor %}
</div>

<a href="{{ "/projects/" | relative_url }}" class="button">
  <i class="fas fa-chevron-circle-right"></i>
  Show More Projects
</a>

<!-- ## Featured <a href="{{ "/publications/" | relative_url }}">Publications</a>

<div class="featured-publications">
  {% assign sorted_publications = site.publications | sort: 'year' | reverse %}
  {% for pub in sorted_publications %}
    {% if pub.highlight %}
      <a href="{{ pub.pdf }}" class="publication">
        <strong>{{ pub.title }}</strong>
        <span class="authors">{% for author in pub.authors %}{{ author }}{% unless forloop.last %}, {% endunless %}{% endfor %}</span>.
        <i>{% if pub.venue %}{{ pub.venue }}, {% endif %}{{ pub.year }}</i>.
        {% for award in pub.awards %}<br/><span class="award"><i class="fas fa-{% if award == "Best Paper" %}trophy{% else %}award{% endif %}" aria-hidden="true"></i> {{ award }}</span>{% endfor %}
      </a>
    {% endif %}
  {% endfor %}
</div>

<a href="{{ "/publications/" | relative_url }}" class="button">
  <i class="fas fa-chevron-circle-right"></i>
  Show All Publications
</a> -->
