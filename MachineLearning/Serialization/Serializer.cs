using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
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
      /// Selettore di surrogati per la serializzazione
      /// </summary>
      public static SurrogateSelector SurrogateSelector { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore statico
      /// </summary>
      static Serializer()
      {
         SurrogateSelector = new SurrogateSelector();
         try {
            var context = new StreamingContext(StreamingContextStates.All);
            var surrogators = Assembly.GetExecutingAssembly().GetTypes()
               .Select(t => (Surrogate: t, Interface: t.GetInterface($"{nameof(ISerializationSurrogate)}`1")))
               .Where(t => t.Interface != null)
               .Select(t => (t.Surrogate, Type: t.Interface.GetGenericArguments()[0]));
            foreach (var s in surrogators) {
               var surrogate = (ISerializationSurrogate)Assembly.GetExecutingAssembly().CreateInstance(s.Surrogate.ToString());
               SurrogateSelector.AddSurrogate(s.Type, context, surrogate);
            }
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
            throw;
         }
      }
      /// <summary>
      /// Clona un oggetto
      /// </summary>
      /// <param name="obj">Oggetto da clonare</param>
      /// <returns>L'oggetto clonato</returns>
      public static T Clone<T>(T obj)
      {
         var formatter = new BinaryFormatter(SurrogateSelector, new StreamingContext(StreamingContextStates.Clone));
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
