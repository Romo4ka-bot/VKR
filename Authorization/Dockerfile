FROM adoptopenjdk/openjdk8

ARG JAR_FILE=vkr.jar

WORKDIR /app

COPY Authorization/target/${JAR_FILE} /app/app.jar

ENTRYPOINT ["java", "-jar", "app.jar"]