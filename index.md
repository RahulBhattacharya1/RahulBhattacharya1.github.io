---
layout: home
title: "Rahul’s Portfolio"
author_profile: true
featured_posts:
  - /intro/post1/
  - /intro/second/
---

<section class="featured-block">
  <h2>Featured</h2>
  <ul class="taxonomy__index">
    {%- assign featured_list = page.featured_posts | compact -%}
    {%- if featured_list and featured_list.size > 0 -%}
      {%- for url in featured_list -%}
        {%- assign post = site.posts | where: "url", url | first -%}
        {%- if post -%}
          <li>
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {%- if post.excerpt -%}<p>{{ post.excerpt | strip_html | truncate: 140 }}</p>{%- endif -%}
          </li>
        {%- endif -%}
      {%- endfor -%}
    {%- else -%}
      {%- for post in site.posts -%}
        {%- if post.featured == true -%}
          <li>
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {%- if post.excerpt -%}<p>{{ post.excerpt | strip_html | truncate: 140 }}</p>{%- endif -%}
          </li>
        {%- endif -%}
      {%- endfor -%}
    {%- endif -%}
  </ul>
</section>
