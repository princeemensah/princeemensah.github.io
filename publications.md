---
layout: page
title: "Publications"
class: publications
permalink: /publications/
---

## Publications

<div class="pubs">
  {% for pub in site.data.publications %}
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
