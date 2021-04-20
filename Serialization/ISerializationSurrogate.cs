using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Interfaccia per i surrogati di serializzazione
   /// </summary>
   /// <typeparam name="T">Il tipo di oggetto da serializzare (da surrogare)</typeparam>
   public interface ISerializationSurrogate<T> : ISerializationSurrogate
   {
   }
}
