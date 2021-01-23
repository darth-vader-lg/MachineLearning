using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace MachineLearning.Serialization
{
#pragma warning disable SYSLIB0011 // Il tipo o il membro è obsoleto
   /// <summary>
   /// Classe helper per la serializzazione
   /// </summary>
   public static class Serializer
   {
      #region Methods
      /// <summary>
      /// Clona un oggetto
      /// </summary>
      /// <param name="obj">Oggetto da clonare</param>
      /// <returns>L'oggetto clonato</returns>
      public static T Clone<T>(T obj)
      {
         var formatter = new BinaryFormatter(new SurrogateSelector(), SurrogateSelector.StreamingContext);
         using var memoryStream = new MemoryStream();
         formatter.Serialize(memoryStream, obj);
         memoryStream.Position = 0;
         return (T)formatter.Deserialize(memoryStream);
      }
      /// <summary>
      /// Deserializza un oggetto o un grafico di oggetti
      /// </summary>
      /// <param name="stream">stream</param>
      /// <param name="formatter">Eventuale formattatore</param>
      /// <returns>L'oggetto od il grafico di oggetti contenuti nello stream</returns>
      public static object Deserialize(Stream stream, IFormatter formatter = null)
      {
         if (stream == null)
            throw new ArgumentException("Parameter cannot be null", nameof(stream));
         formatter ??= new BinaryFormatter(null, SurrogateSelector.StreamingContext);
         var orgSelector = formatter.SurrogateSelector;
         try {
            var surrogatesSelector = new SurrogateSelector();
            surrogatesSelector.SetNextSelector(orgSelector);
            formatter.SurrogateSelector = surrogatesSelector;
            return formatter.Deserialize(stream);
         }
         finally {
            formatter.SurrogateSelector = orgSelector;
         }
      }
      /// <summary>
      /// Serializza un oggetto o un grafico di oggetti
      /// </summary>
      /// <param name="stream">stream</param>
      /// <param name="graph">Grafico o oggetto da serializzare</param>
      /// <param name="formatter">Eventuale formattatore</param>
      public static void Serialize(Stream stream, object graph, IFormatter formatter)
      {
         if (stream == null)
            throw new ArgumentException("Parameter cannot be null", nameof(stream));
         if (graph == null)
            throw new ArgumentException("Parameter cannot be null", nameof(graph));
         formatter ??= new BinaryFormatter(null, SurrogateSelector.StreamingContext);
         var orgSelector = formatter.SurrogateSelector;
         try {
            var surrogatesSelector = new SurrogateSelector();
            surrogatesSelector.SetNextSelector(orgSelector);
            formatter.SurrogateSelector = surrogatesSelector;
            formatter.Serialize(stream, graph);
         }
         finally {
            formatter.SurrogateSelector = orgSelector;
         }
      }
      #endregion
   }
#pragma warning restore SYSLIB0011 // Il tipo o il membro è obsoleto
}
