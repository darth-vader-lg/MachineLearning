using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Classe helper per la serializzazione
   /// </summary>
   public static class Serializer
   {
      #region Properties
      /// <summary>
      /// Surrogato per la serializzazione
      /// </summary>
      public static SurrogateSelector Surrogator { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore statico
      /// </summary>
      static Serializer()
      {
         Surrogator = new SurrogateSelector();
         var context = new StreamingContext(StreamingContextStates.All);
         Surrogator.AddSurrogate(typeof(Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer.Options), context, new SdcaNonCalibratedMulticlassTrainerSerializer.Options.SerializationSurrogate());
         Surrogator.AddSurrogate(typeof(Microsoft.ML.Data.TextLoader.Column), context, new TextLoaderSerializer.Column.SerializationSurrogate());
         Surrogator.AddSurrogate(typeof(Microsoft.ML.Data.TextLoader.Options), context, new TextLoaderSerializer.Options.SerializationSurrogate());
         Surrogator.AddSurrogate(typeof(Microsoft.ML.Data.TextLoader.Range), context, new TextLoaderSerializer.Range.SerializationSurrogate());
      }
      /// <summary>
      /// Clona un oggetto
      /// </summary>
      /// <param name="serializable">Oggetto serializzabile</param>
      /// <returns>L'oggetto clonato</returns>
      public static T Clone<T>(this T obj)
      {
         var formatter = new BinaryFormatter(Surrogator, new StreamingContext(StreamingContextStates.Clone));
         using var memoryStream = new MemoryStream();
#pragma warning disable SYSLIB0011 // Il tipo o il membro è obsoleto
         formatter.Serialize(memoryStream, obj);
         memoryStream.Position = 0;
         return (T)formatter.Deserialize(memoryStream);
#pragma warning restore SYSLIB0011 // Il tipo o il membro è obsoleto
      }
      #endregion
   }
}
