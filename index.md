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

**<span style="color: #ffc000;">I am open to collaboration and actively seeking PhD opportunities. If you're interested in working together or have opportunities available, please email <a href="mailto:princemensah@aims.edu.gh" style="color: #3eb7f0;">me!</a></span>**

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
<!-- * <a href="mailto:princemensah@aims.edu.gh">princemensah@aims.edu.gh</a> -->
</div>

</div>

<div class="news-travel" markdown="1">

<div class="news" markdown="1">
## News

<ul>
{% for item in site.data.news %}
  <li>
    <span class="news-date">{{ item.date | date: "%b, %d, %Y" }}</span>
    <span class="news-text">{{ item.description | markdownify | strip_newlines | remove: '<p>' | remove: '</p>' }}</span>
  </li>
{% endfor %}
</ul>

</div>

</div>

## Publication

<div class="pubs home-pubs">
  {% for pub in site.data.publications limit:3 %}
  <article class="publication">
    <h3>
      {% if pub.url %}
      <a href="{{ pub.url }}" target="_blank" rel="noopener">{{ pub.title }}</a>
      {% else %}
      {{ pub.title }}
      {% endif %}
    </h3>
    <p class="authors">{{ pub.authors }}</p>
    <p class="venue"><em>{{ pub.venue }}</em></p>
    {% if pub.description %}
    <p>{{ pub.description }}</p>
    {% endif %}
    {% if pub.links %}
    <div class="extra-links">
      {% for link in pub.links %}
      <a href="{{ link.url }}" target="_blank" rel="noopener">{{ link.label }}</a>
      {% endfor %}
    </div>
    {% endif %}
  </article>
  {% endfor %}
</div>

<a href="{{ "/publications/" | relative_url }}" class="button">
  <i class="fas fa-chevron-circle-right"></i>
  Show All Publications
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
